import mne
import pyxdf as pyxdf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import  zscore
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from autoreject import AutoReject
import meta_functions_HC as function_hc
from mne.channels import make_standard_montage



#%%

# Information here are provided based on documented protocols and reiz_marker_sa labels
# e.g. based on the protocol we know which hand was stimulated first, based on the reizmarker's labels we know if all the 8 phases and I/O stimuli are saved in the same file
# Below I load the patient data, I have removed the names for privacy reasons. It would look like the name and the stimulated hand

dict_origin_labels  = {
                      
                       '_pre1.xdf': ['L'],
                       '_pre4_old1.xdf': ['L', 'R'],
                       '_post1.xdf': ['L'],
                       '_post3_old1.xdf': ['L'],
                       '_post5_old1.xdf': ['L'],
                       
                       

                       }

                                                                                   #/\      
# Subjects with huge stimulation artifact at the time of trigger (like TMS --------/  \--------)
# The stimulation artifact has been check visually for each participant 
stim_artifact_subs = {'_pre1'  : ['L', [-40, 20]], 
                      }



exdir = "/home/sara/data/Third part/2019_ST_IN-TENS/"
files = list(Path(exdir).glob('**/*.xdf*'))


lists_bad_channels = []
lists_pluses = []
for f in files:
    plt.close('all')
    all_possible_stim_site = dict_origin_labels[f.parts[-1]]
    print(str(f.parts[5]) + '_' + str(f.parts[-1])) # print sb's name
    marker_n = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    C3 = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'C3'][0]
    C3dat = brainvision['time_series'][:,C3]

    
    
    
    
    out = {'pulse_BV':[], 'drop_idx_list': []}
    # bipolar signal near to C3
    #bipolar = pyxdf.load_xdf(f, select_streams=[{'name': 'Spongebob-Data'}])[0][0]
    # pulses creates a list of the indices of the marker timestamps for the stimulation condition trials only
    # recognizing the pulses by checking if the marker value starts with a number 
    pulses = [i for i,m in enumerate(marker_n['time_series']) if marker_n['time_series'][i][0][1].isnumeric()]
    

    

    
    pulses_8_phases = [i for i,m in enumerate(marker_n['time_series']) if "\"phases_to_stimulate\": [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]" in m[0]]
    # deleting the indexes with repeated stimulation marker with a distance of 1, e.g. AmKa marker indexs 11 and 12
    if (np.diff(pulses_8_phases) ==1).any() ==True: 
        pulses_8_phases = np.delete(pulses_8_phases, np.where(np.diff(pulses_8_phases) ==1)[0][0]) # staring point

    

        
        

    

    pulses_0 = [i for i,m in enumerate(marker_n['time_series']) if "\"phases_to_stimulate\": [0.0]" in m[0]]
    if (np.diff(pulses_0) ==1).any() ==True:
        pulses_0 = np.delete(pulses_0, np.where(np.diff(pulses_0) ==1)[0][0]) # staring point
    else:
        pulses_0 = pulses_0    
    if f.parts[-1] == 'TMS_NMES_GuWi_pre4.xdf':
        pulses_0 = pulses_0[1:]   
    elif f.parts[-1] == 'TMS_NMES_BuUl_pre3_old2.xdf':
        pulses_0 = pulses_0[1:]
           
        
    pulses_180 = [i for i,m in enumerate(marker_n['time_series']) if "\"phases_to_stimulate\": [180.0]" in m[0]]
    if (np.diff(pulses_180) ==1).any() ==True:
        pulses_180 = np.delete(pulses_180, np.where(np.diff(pulses_180) ==1)[0][0]) # staring point
    else:
        pulses_180 = pulses_180        
        
        
        
        
    pulses = [i for i,m in enumerate(marker_n['time_series']) if marker_n['time_series'][i][0][1].isnumeric()and i >= pulses_8_phases[0]]    
        
    # pulseinfo contains a list of the stim.condition time stamps and descriptions
    # each item in the list contains a list with the size 2: pulseinfo[i][0] is the timestamp corresponding with the index i from pulses,
    # pulseinfo[i][1] contains the corresponding stimulus description (i.e., stim phase and freq, etc.)
    pulseinfo = [[np.searchsorted(brainvision['time_stamps'], marker_n['time_stamps'][p]), marker_n['time_series'][p]] for p in pulses]
    n=0
    
    
    
    
    # Here we need to set EDC muscle of the right or left hand according to the stimulation site    
    for i, v in enumerate(all_possible_stim_site):
 
        if v == 'R':
            EDC_hand = 'EDC_R'
        elif v == 'L':
            EDC_hand = 'EDC_L'
    
    
        edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == EDC_hand][0]
        edcdat = brainvision['time_series'][:,edcix]
        
        for i,p in enumerate(pulseinfo):
            pulse_idx = pulses[pulseinfo.index(p)]
            sample = p[0]
    
            # For the NMES study, we use the ECD_R data to identify the artifact
            # and we use a time window around the onset of the original reizmarker_timestamp: [sample-1500:sample+1500]
            if sample > 1500:
                onset = sample-1500
                offset = sample+1500
                edcep = edcdat[onset:offset]
                C3ep = C3dat[onset:offset]
                dmy=  np.abs(zscore(edcep))
                dmy_c3  = np.abs(zscore(edcep))
    
                tartifact = np.argmax(dmy)
                tartifact_v = max(dmy) 

    
                # edcep contains 3000 timepoints or samples (-1500 to +1500 samples around the original rm_marker)
                # so, if tartifact is < 1500, the new marker is in the period before the original marker
                # if tartifact is >1500, the new marker is in the period after the original marker      
                corrected_timestamp = sample - 1500 + tartifact
            # these epochs when the preprocessing in MNE is started
            if np.max(dmy) < 3 :
               
                out['drop_idx_list'].append(pulse_idx)
            out['pulse_BV'].append(corrected_timestamp)
        _, _, pulses_ind_drop = np.intersect1d(out['drop_idx_list'], pulses, return_indices=True)
        
        
        marker_corrected = marker_n
        
        for i in range(len(pulses)):
            # for the stim.condition time stamps (corresponding to the indices stored in pulses)
            # replace original reizmarker (rm) timestamp value with the corrected timestamp value based on the EDC artifact (corrected_timestamp)
            rm_timestamp_idx = pulses[i]
            brainvision_idx = out['pulse_BV'][i]
            rm_timestamp_new_value = brainvision['time_stamps'][brainvision_idx] 
                    
            #print('old value: '+str(marker['time_stamps'][pulses[i]]))
            # replace original stimulus onset time stamp with the new timestamp value
            marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
            #print('new value: '+str(marker['time_stamps'][pulses[i]]))
    
            
    
        #### convert brainvision and corrected marker stream into a fif file that can be read by MNE ###    
    
        #marker_corrected = marker    #pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
        data = brainvision   #pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
        marker_corrected['time_stamps'] -= data['time_stamps'][0] #remove clock offset
        
        channel_names = [c['label'][0] for c in data['info']['desc'][0]['channels'][0]['channel'] ]
        sfreq = int(data['info']['nominal_srate'][0])
        types = ['eeg']*64
        types.extend(['emg']*(len(channel_names)-64)) #64 EEG chans, rest is EMG/EKG
        info = mne.create_info(ch_names = channel_names, sfreq = sfreq, ch_types = types)
        raw = mne.io.RawArray(data = data['time_series'][0:int(marker_corrected['time_stamps'][-1]*1500), :].T, info = info)
        
        if len(marker_corrected['time_stamps']) > 1:
            descs = [msg[0] for msg in marker_corrected['time_series']]
            ts = marker_corrected['time_stamps']
        
        sel = [i for i,v in enumerate(descs)  if marker_n['time_series'][i][0][1].isnumeric() and i >= pulses_8_phases[0]]
     
        if len(all_possible_stim_site)==1:
             for i_sel, v_sel in enumerate(sel):
                 if  v_sel in range(pulses_8_phases[0]+1, pulses_0[0], 1):
                    descs[v_sel] = dict_origin_labels[f.parts[-1]][0] + '_' +  descs[v_sel]
                # add "IO" for extra  phases 0 and 180
                 if  v_sel in range(pulses_0[0]+1, sel[-1]+1, 1):
                    descs[v_sel] = dict_origin_labels[f.parts[-1]][0] + '_'  + 'IO'  +  '_' +  descs[v_sel]
                    
                    
                    
                    
                    
                    
                    
        elif len(all_possible_stim_site)>1:
        
            for i_sel, v_sel in enumerate(sel):
                # add "L" or "R" to the labels according to healthy control protocol
                if  v_sel in range(pulses_8_phases[0]+1, pulses_0[0], 1):
                    descs[v_sel] = dict_origin_labels[f.parts[-1]][0] + '_' +  descs[v_sel]
                # add "IO" for extra  phases 0 and 180
                if  v_sel in range(pulses_0[0]+1, pulses_8_phases[1], 1):
                    descs[v_sel] = dict_origin_labels[f.parts[-1]][0] + '_'  + 'IO'  +  '_' +  descs[v_sel]
                    
                if  v_sel in range(pulses_8_phases[1]+1, pulses_0[1], 1):
                    descs[v_sel] = dict_origin_labels[f.parts[-1]][1]    + '_' +  descs[v_sel]
                    
                if  v_sel in range(pulses_0[1]+1, sel[-1]+1, 1):
                    descs[v_sel] = dict_origin_labels[f.parts[-1]][1] + '_' +'IO' +  '_' +  descs[v_sel]
            

        
        descs = [descs[i] for i in sel]        
        ts = [ts[i] for i in sel]


    ts_new = np.delete(ts, pulses_ind_drop)
    shortdescs_new = np.delete(descs, pulses_ind_drop)
    
    
    
    
    
      
    # Find the latency when stimulation switches from one side to the other side
    switch_ind = [i for i,v in enumerate(shortdescs_new)  if v[0]!= dict_origin_labels[f.parts[-1]][0] and v[0]!="'"]

        
   
    for i, v in enumerate(all_possible_stim_site):
        
        if i==0 and len(all_possible_stim_site)>1:
            ts_switch= ts_new[0:switch_ind[0]]
            shortdescs_switch = shortdescs_new[0:switch_ind[0]]
        elif i==1 and len(all_possible_stim_site)>1:    
            ts_switch = ts_new[switch_ind[0]:switch_ind[-1]+1] 
            shortdescs_switch = shortdescs_new[switch_ind[0]:switch_ind[-1]+1]  
        else:
            # To compensate for the cutted file of RaPa subject, TMS_NMES_RaPa_healthy_old1. Just left side is stimulated
             shortdescs_switch =  shortdescs_new 
             ts_switch = ts_new

    
    
    
        anno = mne.Annotations(onset = ts_switch, duration = 0, description = shortdescs_switch)
        raw = raw.set_annotations(anno)  
        if i==0:
            raw.pick_channels(['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Iz','FC1','FC2','CP1',
                                'CP2','FC5','FC6','CP5','CP6','FT9','FT10','TP9','TP10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4','CP3','CP4',
                                'PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz','CPz','POz','Oz',])
 
    
            raw.pick_types(meg = False, eeg = True, ecg = False)
        




        raw._data = mne.filter.notch_filter(raw._data, raw.info['sfreq'], 50, notch_widths =2, phase='zero'  )

        # interpolate channels based on thevariance of the channels 
        raw, badchans  = function_hc.mark_bad_channels_interpolate(raw)
        # raw.plot_psd(tmax=250, average=False)
        
        
        
        # Creating epochs
        (events_from_annot, event_dict) = mne.events_from_annotations(raw)
        u, indices = np.unique(events_from_annot[:,0], return_index=True)
        events_from_annot_unique = events_from_annot[indices]
        event_unique, event_unique_ind  = np.unique(events_from_annot_unique[:,2], return_index=True)
        
        # Create epochs based on the events, from -1 to 1s
        # Set the baseline to None, because mne suggests to do a baseline correction after ICA
        epochs = mne.Epochs(raw, events_from_annot_unique, event_id=event_dict,
                            tmin=-1, tmax=1, reject=None, preload=True,  baseline=None, on_missing = 'ignore')
        

     
        
        # cubic interpolation for some subjects that have sth like a TMS artifact
        for sub_name, sub_name_v in enumerate(stim_artifact_subs):   
            if (f.parts[-1][9:18] == sub_name_v):
                for l_stim_art in range(len(stim_artifact_subs[f.parts[-1][9:18]])): 
                    if (f.parts[-1][9:18] == sub_name_v and stim_artifact_subs[f.parts[-1][9:18]][l_stim_art] ==  v )==True:
                        epochs_interpolated = function_hc.cubic_interp(epochs, win = stim_artifact_subs[f.parts[-1][9:18]][1])
            else:
                epochs_interpolated = epochs







        # Filtering
        # bandstop (48-52 Hz) filter data. lfreq higher than hfreq makes a bandstop filter
        epochs_interpolated._data = mne.filter.notch_filter(epochs_interpolated._data, epochs.info['sfreq'], 50, notch_widths=2, phase='zero', verbose=0, pad = 'constant')
        epochs_interpolated.filter(1, 45, method='iir', verbose=0, iir_params=None, pad = 'reflect_limited') # If iir_params is none for iir filter then it will consider butterworth of order 4
        evokeds = epochs_interpolated.average()
        evokeds.plot(spatial_colors = True, gfp=True)
        
        
        
        
     
        ar = AutoReject(n_interpolate=[0])
        epochs_ar, reject_log_1 = ar.fit(epochs_interpolated).transform(epochs_interpolated, return_log = True)   
        #Applying ICA after filtering and before baseline correction 
        data_ica  = function_hc.clean_dataset(epochs_ar)
        epochs_cleaned = data_ica['eeg']
        epochs_b = epochs_cleaned.apply_baseline(baseline=(-0.9, -0.1))
        evokeds = epochs_b.average()
        

        all_times = np.arange(0, 0.8, 0.02)
        fig_topo = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
        fig_erp = evokeds.plot(spatial_colors = True, gfp=True)
        fig_erp.set_size_inches((20, 8))
        plt.plot(raw._data[4,:])
        plt.show()
        
        
        # Save epoch files and figures
        save_folder = "/home/sara/data/Third part/epochs/"
        save_folder_figs = "/home/sara/data/Third part/epochs/figs/"
        epochs_b.save(save_folder + str(f.parts[-1][9:-4]) + v +'_epo.fif', overwrite = True, split_size='2GB')
        fig_topo.savefig(save_folder_figs +  str(f.parts[-1][9:-4]) + '_' + v + '_topo' + '.svg')
        fig_erp.savefig(save_folder_figs +  str(f.parts[-1][9:-4]) + '_' + v + '_erp'  + '.svg')

    



