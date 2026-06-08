from pathlib import Path

import mne
import math
import numpy as np
from scipy import stats
#import pickle
import itertools  
import mne.stats
from tqdm import tqdm
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.stats import  zscore
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt 
import pandas as pd
from mne.stats import permutation_cluster_test
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from scipy.signal import medfilt
import scipy.stats 
#import meta_functions_hc as function_hc
import pickle




def cubic_interp(epochs, win):

    from scipy import interpolate

    # convert interpolation window to indices in epochs
    idx1 = np.argmin(np.abs(epochs.times - win[0]*0.001))
    idx2 = np.argmin(np.abs(epochs.times - win[1]*0.001))

    # get timepoints in seconds and delete those that should be interpolated
    x = epochs.times
    x = np.delete(x, np.s_[idx1:idx2], 0)
    

    for i, epoch in enumerate(epochs):

        y = np.delete(epoch, np.s_[idx1:idx2], -1)
        p = interpolate.interp1d(x, y, kind='cubic')

        # get the interpolation values for the timepoints of interest
        interp_values = p.__call__(epochs.times[idx1:idx2])

        # for each epoch replace timepoints in epochs object
        epochs._data[i, :, idx1:idx2] = interp_values

    return epochs


def clustering_channels(n_sub, win_erps, exdir_epoch, thresholds, labels, name, save_folder):
    
   
    files = Path(exdir_epoch).glob('*epo.fif*')
    #plt.close('all')
    unique_phases = np.arange(0, 360, 45)
    stim_int = np.arange(2, 18, 2)
    peaks = np.zeros([n_sub, 64, len(labels), len(unique_phases), len(stim_int)])     
    a = np.zeros([64, len(labels)])

    

    
    for ifiles, f in enumerate(files):
        print(ifiles, f)

        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        epochs_p1 =  mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        # removing the effect of phase amp according to Granö et al. 2022.
        # amp after stim - amp before stim     
        epochs_amp_mod = epochs._data[:,:,1001:] - epochs._data[:,:,0:1000] # to correct phase estimation
        # making mne epoch structure
        epochs = mne.EpochsArray(data = epochs_amp_mod,  info = epochs.info, events = epochs.events, event_id = epochs.event_id, on_missing='ignore')

        #non_central_ch = flipping_ch(epochs.info['ch_names'])  
        epochs_bystimandphase = {} 
        epochs_bystimandphase_p1 = {}
        erp_bystimandphase = {} 
        peaks_bystimandphase = {}
        erp_bystimandphase_p1 = {}
     
        
        for istim, stim in enumerate(stim_int):
            epochs_bystimandphase_p1[str(stim)] = {}
            epochs_bystimandphase[str(stim)] ={}
            erp_bystimandphase[str(stim)] = {} 
            erp_bystimandphase_p1[str(stim)] = {}
            peaks_bystimandphase[str(stim)] = {} 
          
            for iphase, phase in enumerate(unique_phases):
                sel_idx = Select_Epochs_intensity_phase(epochs, stim, phase)
                epochs_bystimandphase_p1[str(stim)][str(phase)] = epochs_p1[Select_Epochs_intensity_phase(epochs_p1, stim, phase)] # This is because of stim artifact around 0
                erp_bystimandphase_p1[str(stim)][str(phase)]  = epochs_bystimandphase_p1[str(stim)][str(phase)].average() 
                epochs_bystimandphase[str(stim)][str(phase)] = epochs[sel_idx]
                erp_bystimandphase[str(stim)][str(phase)]  = epochs_bystimandphase[str(stim)][str(phase)].average() 

                        
                for ipeak, peak in enumerate(labels):
              
                    
                    if ipeak == 0:    #P45
                        if (f.parts[-2] == 'v2' and f.parts[-3] == 'Left'):
                            peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase_p1[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                        else:
                            peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[0,0]:win_erps[0,1]]),1)
                    
                    elif  ipeak == 1: #N60
                            peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[1,0]:win_erps[1,1]]),1)
      
                    elif  ipeak == 2: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[2,0]:win_erps[2,1]]),1)

                    elif  ipeak == 3: 
                        peaks_bystimandphase[str(stim)][str(phase)] = np.mean((erp_bystimandphase[str(stim)][str(phase)]._data[:,win_erps[3,0]:win_erps[3,1]]),1)


                    if str(erp_bystimandphase[str(stim)][str(phase)].comment) == str(''):    # To remove none arrays after selecting epochs
                        peaks_bystimandphase[str(stim)][str(phase)] = np.zeros(64) 
                       
                    else:
                        peaks[ifiles, :, ipeak, iphase, istim] = (peaks_bystimandphase[str(stim)][str(phase)] )
             
    

              
                

    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs.info , 'eeg')
    clusters, mask, pvals = permutation_cluster(peaks, adjacency_mat, thresholds)         
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
        
    #combine labels 2 and 3, they are the same component. just different cluster    
    #a_com = a[:,[0, 1, 2, 4]]
    a[a > 4] = 4
    a[a < -4] = -4
    if name == 'R_v6':
        a[:,2][a[:,2] > 3] = 2 # to remove nocise related to ocular channels. 
    fig = plot_topomap_peaks_second_v(name, a, mask, ch_names, pvals,[-5,5], epochs.info, i_intensity = 'all')
   
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
        

    fig.savefig(save_folder + name + '.svg')    

    return all_ch_names_biggets_cluster, pvals, a, mask, epochs.info, np.mean(peaks, (-2, -1))










def channel_names():
    ch_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Iz','FC1','FC2','CP1',
    'CP2','FC5','FC6','CP5','CP6','FT9','FT10','TP9','TP10','F1','F2','C1','C2','P1','P2','AF3','AF4','FC3','FC4','CP3','CP4',
    'PO3','PO4','F5','F6','C5','C6','P5','P6','AF7','AF8','FT7','FT8','TP7','TP8','PO7','PO8','Fpz','CPz','POz','Oz']
    return ch_names



def permutation_cluster(peaks, adjacency_mat, thresholds):
    #from sklearn import preprocessing
    
    # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    

    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    mask = np.zeros([nchans, npeaks])
    pvals = np.zeros([npeaks])
    clusters = []
   

    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        if p == 0:
            mean_peaks[:, [63, 9, 59, 15, 29, 8, 58, 19, 14], ] = 0.006 # control group, P40 cluster, very noisy channel


        cluster = mne.stats.permutation_cluster_1samp_test(zscore((mean_peaks[:,:,p]), axis =1), out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        

            
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
       
        
        
        if len(t_sum) > 0:
                mask[:,p] = cluster[1][np.argmax(t_sum)]
                pvals[p] = min(cluster[2])
            

        clusters.append(cluster)         
        

    return clusters, mask, pvals


def plot_topomap_peaks_second_v(name, peaks, mask, ch_names, pvals, clim, pos, i_intensity):

   
    nplots =1 
    nchans, npeaks = np.shape(peaks)

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks, figsize=(10,6))
    plt.style.use('default')
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            # if mask is None:
            #     psig = None
            # else:
            #     psig = np.where(mask[iplot, :, ipeak] < 0.01, True, False)

            # sps[ipeak, iplot].set_aspect('equal')

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d(ch_names, peaks[ :, ipeak], pos,
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    if i_intensity == 'all':
        fig.suptitle('All Intensities and Phases Vs zero_' + f'{name}', fontsize = 14)
    else:
        fig.suptitle(f'{ (i_intensity+1)*2} mA', fontsize = 14)
    #fig.suptitle('All Frequencies and all phases', fontsize = 14)
# =============================================================================
#     sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n  cluster_pv = {pvals[0]}')
#     sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {pvals[1]}')
# =============================================================================
    sps[0].set_title('\n\n P1' , fontsize=14, fontweight ='bold')
    sps[1].set_title('\n\n N1' , fontsize=14, fontweight ='bold')
    sps[2].set_title('\n\n N2',  fontsize=14, fontweight ='bold')
    sps[3].set_title('\n\n P2',  fontsize=14, fontweight ='bold')
    #sps[3].set_title('\n\n P200', fontsize=14, fontweight ='bold')
    if pvals is not None:
        fig.text(0.17, 0.3, f' P = {np.round(pvals[0], 3)} ',  ha='left',fontsize=14)
        fig.text(0.34, 0.3, f' P = {np.round(pvals[1], 3)} ',  ha='left', fontsize=14)
        fig.text(0.55,  0.3, f' P = {np.round(pvals[2], 3)} ',  ha='left',fontsize=14)
        fig.text(0.75, 0.3, f' P = {np.round(pvals[3], 3)} ',  ha='left',fontsize=14)
        #fig.text(0.73, 0.3, f' P = {np.round(pvals[3], 2)} ',  ha='left', fontsize=14)
    
    #fig.text(0, 0. ,f' \n\n  TH = {thresholds[0]} \n\n  cluster_pv = {pvals_all[str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(0)][str(1)]}\n\n  ',  ha='left')
    #fig.text(0.5, 0 ,f' \n\n  TH = {thresholds[1]} \n\n  cluster_pv = {pvals_all[str(1)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(0)]}\n\n {all_ch_names_biggets_cluster[str(1)][str(1)]}\n\n  ',  ha='left')
    cb.set_label('t-value', rotation = 90)

    

    plt.show()

    return fig


def topoplot_2d (ch_names, ch_attribute, pos, clim=None, axes=None, mask=None, maskparam=None):
    
    """
    Function to plot the EEG channels in a 2d topographical plot by color coding 
    a certain attribute of the channels (such as PSD, channel specific r-squared).
    Draws headplot and color fields.
    Parameters
    ----------
    ch_names : String of channel names to plot.
    ch_attribute : vector of values to be color coded, with the same length as the channel, numerical.
    clim : 2-element sequence with minimal and maximal value of the color range.
           The default is None.
           
    Returns
    -------
    None.
    This function is a modified version of viz.py (mkeute, github)
    """    

    import mne
    # get standard layout with over 300 channels
    #layout = mne.channels.read_layout('EEG1005')
    
    # select the channel positions with the specified channel names
    # channel positions need to be transposed so that they fit into the headplot
# =============================================================================
#     pos = (np.asanyarray([layout.pos[layout.names.index(ch)] for ch in ch_names])
#            [:, 0:2] - 0.5) / 5
#     
# =============================================================================
    if maskparam == None:
        maskparam = dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                    linewidth=0, markersize=3) #default in mne
    if clim == None:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  ch_type='eeg',
                                  sensors=True,
                                  contours=5,
                                  cmap = 'RdBu_r',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vlim = (clim[0], clim[1]),
                                  sphere=(0.00, 0.00, 0.00, 0.11),
                                  extrapolate = 'head')
    else:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  ch_type='eeg',
                                  sensors=True,
                                  contours=5,
                                  cmap = 'RdBu_r',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vlim = (clim[0], clim[1]),
                                  sphere=(0.00, 0.00, 0.00, 0.11),
                                  extrapolate = 'head')
    return im





def Select_Epochs_intensity_phase(epochs, stim, phase):
    """ 
    this is a function that will identify epochs based on their key (a string) in event_id, 
    which describes the stimulation condition
        
    selection depends on the frequency and the phase of interest
        
    the function returns a list of event indices, that only includes the indices of epochs that contained 
    stimulation at the desired frequency and phase
        
        
    data: epochs data in MNE format
    freq: an integer number, this can be any number between 0 and 40 and depends on the frequencies
    that were stimulated in your study (and thus described in your event description (a string) in event_id)
    phase: an integer number, this can be any number between 0 and 360 and depends on the phases
    that were stimulated in your study (and thus described in your event description in event_id)
    """
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    stim_to_select = str(stim) 
    phase_to_select = str(phase) 
    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if event_id_key.find("I") == -1: # To exclude IO events (extra 0 and 180 pulses)
            if phase >= 0 and phase <=360:
                #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
                if (stim_to_select == str(event_id_key[event_id_key.find('m') - (event_id_key.find('m')-event_id_key.find("'")-1):event_id_key.find('m')]))  and (phase_to_select == str(int(float(event_id_key[event_id_key.find('m') + 4:])))) :    
                    index_list.append(i)      
                else:
                    continue

    return index_list
    
   

def laterality_Index_errorbar(contra_right, contra_left, evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, save_folder_peak):

    contra_right_ind = channel_indices(contra_right)
    contra_left_ind = channel_indices(contra_left)
    contra_stroke = np.zeros([10, 315]); ipsi_stroke = np.zeros([10, 315])   
    contra_hc = np.zeros([17, 315]); ipsi_hc = np.zeros([17, 315])   
    
    n1 = 985
    # Stroke ERPs
    for i, _ in enumerate(evokeds_all_L[str('v2')]):     
        contra_stroke[i, :] = np.mean(evokeds_all_L[str('v2')][i]._data[contra_left_ind, n1:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R[str('v2')]):
        contra_stroke[i+6, :] = np.mean(evokeds_all_R[str('v2')][i]._data[contra_right_ind, n1:1300], axis = 0)
    for i, _ in enumerate(evokeds_all_L[str('v2')]):     
        ipsi_stroke[i, :] = np.mean(evokeds_all_L[str('v2')][i]._data[contra_right_ind, n1:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R[str('v2')]):
        ipsi_stroke[i+6, :] = np.mean(evokeds_all_R[str('v2')][i]._data[contra_left_ind, n1:1300], axis = 0)
        
    # Healthy ERPs
    for i, _ in enumerate(evokeds_all_L_hc):     
        contra_hc[i, :] = np.mean(evokeds_all_L_hc[i]._data[contra_left_ind, n1:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R_hc):
        contra_hc[i+11, :] = np.mean(evokeds_all_R_hc[i]._data[contra_right_ind, n1:1300], axis = 0)
    for i, _ in enumerate(evokeds_all_L_hc):     
        ipsi_hc[i, :] = np.mean(evokeds_all_L_hc[i]._data[contra_right_ind, n1:1300], axis = 0)    
    for i, _ in enumerate(evokeds_all_R_hc):
        ipsi_hc[i+11, :] = np.mean(evokeds_all_R_hc[i]._data[contra_left_ind, n1:1300], axis = 0)    
        
        
        
    LI_ST = (contra_stroke - ipsi_stroke) / (contra_stroke + ipsi_stroke)  
    LI_HC = (contra_hc - ipsi_hc) / (contra_hc + ipsi_hc)  
    
    
    LI_ST[LI_ST > 50] = 50 # not  condisering outliers
    LI_ST[LI_ST < -50] = -50
    
    LI_HC[LI_HC > 50] = 50
    LI_HC[LI_HC < -50] = -50
    
    
    
    
    
    
    fig, ax = plt.subplots()
    ax.set_title('N120 Laterality Index',  fontweight="bold",    fontsize =12)
    data_std = np.zeros([1, 2])
    data_mean =  np.zeros([1, 2])
    start_time = 115
    end_time = 150
    
    data_mean[0, 0] = np.mean(np.mean(LI_HC[:, start_time:end_time], axis =1), axis =0)
    data_mean[0, 1] = np.mean(np.mean(LI_ST[:, start_time:end_time], axis =1), axis =0)  
    
    data_std[0, 0] = np.std(np.mean(LI_HC[:, start_time:end_time], axis =1), axis =0)/np.sqrt(17)
    data_std[0, 1] = np.std(np.mean(LI_ST[:, start_time:end_time], axis =1), axis =0)/np.sqrt(10)
    
    t_ind , p_ind = stats.ttest_ind(np.mean(LI_HC[:, start_time:end_time], axis =1), np.mean(LI_ST[:, start_time:end_time], axis =1))
    ax.text(0.3, 2.5, f'P = {np.round(p_ind, 3)}', fontweight = 'bold', fontsize=12)
    ax.scatter(0.5, 2.2, marker = '*', color = 'k')
    ax.plot([0, 1], [2, 2], color = 'k')
    ax.errorbar(x = [0,1], y = data_mean[0], yerr= data_std[0],  color ='k', fmt = '.', capsize=10, capthick=3, elinewidth=2)
    ax.set_xticklabels([  '','', 'C',  '',   '', '','S-V2', ''], fontsize =12)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-3, 3])
    ax.set_ylabel('Laterality Index', fontsize =14)
    plt.show()
    fig.savefig(save_folder_peak  + 'Laterality_Index_errorbar'+ '.svg', overwrite = True)
    return(fig)

def channel_indices(ch_names):
    ch_inds = []
    for ch in ch_names:
        ch_ind = [i for i,v in enumerate(channel_names()) if v == ch][0]
        ch_inds.append(ch_ind)
    return(ch_inds)



    

    
def amp_latency_6_components_st(evokeds_all_L, evokeds_all_R, contra_right, contra_left, time_windows, save_folder_peak):    
    amp_r = {}
    lat_r = {}
    amp_l = {}
    lat_l = {}
    amp_lat_r = {}
    amp_lat_l = {}
    
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']       
    for i_r,_ in enumerate(range(len(evokeds_all_R[str('v2')]))):
        
        amp_r[str(i_r)] = np.zeros([len(time_points),len(time_windows)])
        lat_r[str(i_r)] = np.zeros([len(time_points),len(time_windows)])
    
        if i_r== 0:
            title_r = 'BuUl(R)_ipsi'; i_corrected = [3, 0, 3, 0, 0]
        elif i_r== 1:
            title_r = 'EiHe(R)_contra'; i_corrected = [0, 2, 1, 1, 2]
        elif i_r==2:
            title_r = 'GuWi(R)_ipsi'; i_corrected = [2, 1, 2, 3, 3]
        elif i_r ==3:
            title_r = 'MeRu(R)_contra'; i_corrected = [1, 3, 0, 2, 1]
            
            
        
        
    
        amp_lat_r[str(i_r)] = {}
        for i_time_point, time_point in enumerate(time_points):

            
            amp_lat_r[str(i_r)][str(time_point)] = {}
            if time_point == 'v2':
                x = 0; y = 0 ; title = f'{time_point}'
            elif time_point == 'v3':
                x = 0; y = 1 ; title = f'{time_point}'
            elif time_point == 'v4':
                x = 0; y = 2 ; title = f'{time_point}'
            elif time_point == 'v5':
                x = 1; y = 0 ; title = f'{time_point}'
            elif time_point == 'v6':
                x = 1; y = 1 ; title = f'{time_point}'
            
            for i_contra_right, n_contra_right in enumerate(contra_right):  

    
                amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)] = np.zeros([len(contra_right), 1001])
                contra_right_ind = channel_indices(contra_right[str(n_contra_right)])
                a = zscore(evokeds_all_R[str(time_point)][i_corrected[i_time_point]].data[:, ], axis =0)
                amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)] = np.mean(a[contra_right_ind, 1000:], axis = 0) 
                
                
                if i_contra_right == 0:
                    color = 'g'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    
                elif i_contra_right == 1:
                    color = 'g'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.min(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                elif i_contra_right == 2:
                    color = 'navy'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.min(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                    
                elif i_contra_right == 3:
                    color = 'maroon'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    
                elif i_contra_right == 4:
                    color = 'royalblue'
                    amp_r[str(i_r)][i_time_point, i_contra_right] = np.max(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                    lat_r[str(i_r)][i_time_point, i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(time_point)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]
    

    
    
    
    #############################
    # Left Side
    
    for i_l,_ in enumerate(range(len(evokeds_all_L[str('v2')]))):
        
        amp_l[str(i_l)] = np.zeros([len(time_points),len(time_windows)])
        lat_l[str(i_l)] = np.zeros([len(time_points),len(time_windows)])
    
            
        if   i_l == 0:
            title_l = 'AmWo(L)_contra'; i_corrected = [2, 5, 4, 4, 4]
        elif i_l == 1:
            title_l = 'FuMa(L)_ipsi'; i_corrected = [3, 0, 5, 5, 5]
        elif i_l == 2:
            title_l = 'GrMa(L)_ipsi'; i_corrected = [4, 2, 3, 2, 3]
        elif i_l == 3:
            title_l = 'KaBe(L)_contra'; i_corrected = [0, 4, 0, 3, 1]
        elif i_l == 4:
            title_l = 'SoFa(L)_contra'; i_corrected = [5, 3, 2, 0, 2]
        elif i_l == 5:
            title_l = 'WiLu(L)_ipsi'; i_corrected = [1, 1, 1, 1, 0]
        
        
    
        amp_lat_l[str(i_l)] = {}
        for i_time_point, time_point in enumerate(time_points):
            
            amp_lat_l[str(i_l)][str(time_point)] = {}
            if time_point == 'v2':
                x = 0; y = 0 ; title = f'{time_point}'
            elif time_point == 'v3':
                x = 0; y = 1 ; title = f'{time_point}'
            elif time_point == 'v4':
                x = 0; y = 2 ; title = f'{time_point}'
            elif time_point == 'v5':
                x = 1; y = 0 ; title = f'{time_point}'
            elif time_point == 'v6':
                x = 1; y = 1 ; title = f'{time_point}'
            
            for i_contra_left, n_contra_left in enumerate(contra_left):  
                
    
                amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)] = np.zeros([len(contra_left), 1001])
                contra_left_ind = channel_indices(contra_left[str(n_contra_left)])
                a = zscore(evokeds_all_L[str(time_point)][i_corrected[i_time_point]].data[:, ], axis =0)
                amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)] = np.mean(evokeds_all_L[str(time_point)][i_corrected[i_time_point]].data[contra_left_ind, 1000:], axis = 0) 
        
                
    
                
                if i_contra_left == 0:
                    color = 'g'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                    
                elif i_contra_left == 1:
                    color = 'g'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.min(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                elif i_contra_left == 2:
                    color = 'navy'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.min(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                    
                elif i_contra_left == 3:
                    color = 'maroon'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    
                elif i_contra_left == 4:
                    color = 'royalblue'
                    amp_l[str(i_l)][i_time_point, i_contra_left] = np.max(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                    lat_l[str(i_l)][i_time_point, i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(time_point)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]
    

    
    
    ### 
    
    
    lat_t_std = np.zeros([len(lat_l[str(0)]),len(time_windows)])        
    lat_t_m = (lat_l[str(0)]+ lat_l[str(1)]+ lat_l[str(2)]+ lat_l[str(3)]+lat_l[str(4)]+ lat_l[str(5)] + lat_r[str(0)] + lat_r[str(1)] + lat_r[str(2)] + lat_r[str(3)])/10   
    for i_components, n_components in enumerate(time_windows):  
        lat_t_std[:, i_components] = np.std((lat_l[str(0)][:, i_components],lat_l[str(1)][:, i_components], lat_l[str(2)][:, i_components], lat_l[str(3)][:, i_components], lat_l[str(4)][:, i_components], lat_l[str(5)][:, i_components], lat_r[str(0)][:, i_components], lat_r[str(1)][:, i_components], lat_r[str(2)][:, i_components], lat_r[str(3)][:, i_components]), axis=0)        
    
    fig = plt.figure()    
    for i_components, n_components in enumerate(time_windows): 
       
        if i_components == 0:
            color = 'b'
            
        elif i_components == 1:
            color = 'r'
            
        elif i_components == 2:
            color = 'navy'
            
        elif i_components == 3:
            color = 'maroon'
         
        elif i_components == 4:
            color = 'royalblue'
                         
 
        plt.plot(range(len(time_points)),  lat_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
        plt.errorbar(range(len(time_points)),  lat_t_m[:,i_components], lat_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
        plt.xticks(range(len(time_points)), time_points, fontsize=12)
        plt.xlabel('Time points')
        plt.ylabel('Latency')
        plt.title('Both sides')
        plt.legend()
        plt.show()
        
    
        
    fig.savefig(save_folder_peak +  'Latency_6components' + '.svg', overwrite = True) 
    
        
    
        
        
    amp_t_std = np.zeros([len(lat_r[str(0)]),len(time_windows)])        
    amp_t_m= np.zeros([len(lat_r[str(0)]),len(time_windows)])        
    
    for i_components, n_components in enumerate(time_windows):  
        amp_t_std[:, i_components] =np.std((amp_l[str(0)][:, i_components], amp_l[str(1)][:, i_components], amp_l[str(2)][:, i_components], amp_l[str(3)][:, i_components], amp_l[str(4)][:, i_components], amp_l[str(5)][:, i_components], amp_r[str(0)][:, i_components],amp_r[str(1)][:, i_components], amp_r[str(2)][:, i_components], amp_r[str(3)][:, i_components]), axis=0)        
        amp_t_m[:, i_components] = np.mean((amp_l[str(0)][:, i_components], amp_l[str(1)][:, i_components], amp_l[str(2)][:, i_components], amp_l[str(3)][:, i_components], amp_l[str(4)][:, i_components], amp_l[str(5)][:, i_components], amp_r[str(0)][:, i_components],amp_r[str(1)][:, i_components], amp_r[str(2)][:, i_components], amp_r[str(3)][:, i_components]), axis=0)             
    fig = plt.figure()    
    for i_components, n_components in enumerate(time_windows): 
       
        if i_components == 0:
            color = 'b'
            
        elif i_components == 1:
            color = 'r'
            
        elif i_components == 2:
            color = 'navy'
            
        elif i_components == 3:
            color = 'maroon'
         
        elif i_components == 4:
            color = 'royalblue'
                         

        plt.plot(range(len(time_points)),  amp_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
        plt.errorbar(range(len(time_points)),  amp_t_m[:,i_components], amp_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
        plt.xticks(range(len(time_points)), time_points, fontsize=12)
        plt.xlabel('Time points')
        plt.ylabel('Amplitude')
        plt.title('Both Sides')
        plt.legend()
        plt.show()
        
        
        
    fig = plt.figure()    


    plt.plot(range(len(time_points)),  amp_t_m[:,i_components],  color = f'{color}', label = f'{n_components}', alpha = 0.5)
    plt.errorbar(range(len(time_points)),  amp_t_m[:,i_components], amp_t_std[:,i_components], color = f'{color}',  linestyle='None', marker='o')
    plt.xticks(range(len(time_points)), time_points, fontsize=12)
    plt.xlabel('Time points')
    plt.ylabel('Amplitude')
    plt.title('Both Sides')
    plt.legend()
    plt.show()
    fig.savefig(save_folder_peak +  'Amplitude_6components' + '.svg', overwrite = True) 
    
    
    amp_l_arr = np.zeros([5, 6, 5])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            amp_l_arr[i_components,i,:] = amp_l[str(i)][:,i_components]
    
    
    lat_l_arr = np.zeros([5, 6, 5])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            lat_l_arr[i_components,i,:] = lat_l[str(i)][:,i_components]
    
    
    amp_r_arr = np.zeros([5, 4, 5])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            amp_r_arr[ i_components,i,:] = amp_r[str(i)][:,i_components]
            
    lat_r_arr = np.zeros([5, 4, 5])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            lat_r_arr[ i_components,i,:] = lat_r[str(i)][:,i_components]
            
    return(amp_l_arr, amp_r_arr, lat_l_arr, lat_r_arr)        




def box_plot_p1_all_time_points_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_p1 = {'Control': lat_t_hc[component_number, :], 'Stroke V2': np.concatenate((lat_r_arr_st[component_number, :, 0], lat_l_arr_st[component_number, :, 0])),
                                                         'Stroke V3': np.concatenate((lat_r_arr_st[component_number, :, 1], lat_l_arr_st[component_number, :, 1])),
                                                         'Stroke V4': np.concatenate((lat_r_arr_st[component_number, :, 2], lat_l_arr_st[component_number, :, 2])),
                                                         'Stroke V5': np.concatenate((lat_r_arr_st[component_number, :, 3], lat_l_arr_st[component_number, :, 3])), 
                                                         'Stroke V6': np.concatenate((lat_r_arr_st[component_number, :, 4], lat_l_arr_st[component_number, :, 4]))}
  
    
    
    
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_p1[str(j)]) for i,j in enumerate(list(dict_p1.keys()))]
    data_mean = [np.mean(dict_p1[str(j)]) for i,j in enumerate(list(dict_p1.keys()))]
    fig, ax = plt.subplots()
    ax.plot(np.array([1, 2, 3, 4, 5]), data_mean[1:], color = 'k')
    ax.bar( x = np.arange(len(dict_p1.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['C','C', 'S-V2', 'S-V3', 'S-V4', 'S-V5', 'S-V6'], fontsize=12) 
    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.set_title('P30 Latency', fontweight="bold")
    t_ind , p_ind = stats.ttest_ind(dict_p1[str('Control')], dict_p1[str('Stroke V2')])
    t_paired, p_paired = stats.ttest_rel(dict_p1[str('Stroke V2')], dict_p1[str('Stroke V5')])
    ax.text(-.1, 65, f'P = {np.round(p_ind, 3)}', fontweight = 'bold', fontsize = 12)
    t_paired, p_paired = stats.ttest_rel(dict_p1[str('Stroke V2')], dict_p1[str('Stroke V6')])
    ax.text(2. , 65, f'P = {np.round(p_paired, 3)}', fontweight = 'bold', fontsize = 12)
    ax.scatter(.55, 58, marker = '*', color = 'k')
    ax.plot([0, 1], [55, 55], color = 'k')
    ax.plot([0, 0, 0, 0], [55, 54, 53, 52], color = 'k')
    ax.plot([1, 1, 1, 1], [55, 54, 53, 52], color = 'k')
    ax.plot([1, 2, 3, 4, 5], [60, 60, 60, 60, 60], color = 'k')
    ax.plot([1, 1, 1, 1], [60, 59, 58, 57], color = 'k')
    ax.plot([5, 5, 5, 5], [60, 59, 58, 57], color = 'k')
    ax.set_ylim([0, 70])
    plt.show()
    fig.savefig(save_folder_peak  + 'p30_latency_barplot'+ '.svg', overwrite = True)

    return(p_ind, fig)





def N3_cluster_bilaterality(t_r, t_l, pvals_all_r, pvals_all_l, mask_r, mask_l, pos, t_r_hc, t_l_hc, pvals_all_r_hc, pvals_all_l_hc, mask_r_hc, mask_l_hc, save_folder_peak):

   
    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=5)
    fig, sps = plt.subplots(nrows=2, ncols=6, figsize=(20,8))
    plt.style.use('default')
    time_points = ['v2', 'v3', 'v4', 'v5', 'v6']
    time_points_capital = ['V2', 'V3', 'V4', 'V5', 'V6']
    
    for iplot in range(5):
        for ipeak, time_point in enumerate(time_points):
            imask = mask_r[str(time_point)][:,2]
            im = topoplot_2d(channel_names(), t_r[str(time_point)][:,2], pos,
                                clim=[-5,5], axes=sps[0,ipeak], mask=imask, maskparam=maskparam)
            sps[0,ipeak].set_title(f'{time_points_capital[ipeak]}', fontweight = 'bold', fontsize = 20)
            #sps[0, ipeak].text(-0.05, -0.2 , f'p = {np.round(pvals_all_r[str(time_point)][0], 3)}', fontweight = 'bold', fontsize = 14)
            sps[0, ipeak].text(-0.05, -0.2 , f't = {np.round( sum(t_r[str(time_point)][np.where(mask_r[str(time_point)][:, 2] == 1)[0]  , 2]),2)}', fontweight = 'bold', fontsize = 14)
            
            
    for iplot in range(5):
        for ipeak, time_point in enumerate(time_points):
            imask = mask_l[str(time_point)][:,2]
            im = topoplot_2d(channel_names(), t_l[str(time_point)][:,2], pos,
                                clim=[-5,5], axes=sps[1,ipeak], mask=imask, maskparam=maskparam)       
            sps[1,ipeak].set_title(f'{time_points_capital[ipeak]}', fontweight = 'bold', fontsize = 20)
            #sps[1, ipeak].text(-0.05, -0.2 , f'p = {np.round(pvals_all_l[str(time_point)][0], 3)}', fontweight = 'bold', fontsize = 14)
            sps[1, ipeak].text(-0.05, -0.2 , f't = {np.round(sum(t_l[str(time_point)][np.where(mask_l[str(time_point)][:, 2] == 1)[0]  , 2]), 2)}', fontweight = 'bold', fontsize = 14)
    
    
    
    topoplot_2d(channel_names(), t_r_hc[:, 2], pos, clim=[-5,5], axes=sps[0,5], mask=mask_r_hc[:, 2], maskparam=maskparam) 
    sps[0,5].set_title('C', fontweight = 'bold', fontsize = 20)
    #sps[0,5].text(-0.05, -0.2 , f'p = {np.round(pvals_all_r_hc[0], 3)}', fontweight = 'bold', fontsize = 14)
    sps[0,5].text(-0.05, -0.2 , f't = {np.round(sum(t_r_hc[np.where(mask_r_hc[:, 2] ==1)[0], 2]), 2)}', fontweight = 'bold', fontsize = 14)
    
    topoplot_2d(channel_names(), t_l_hc[:, 2], pos, clim=[-5,5], axes=sps[1,5], mask=mask_l_hc[:, 2], maskparam=maskparam) 
    sps[1,5].set_title('C', fontweight = 'bold', fontsize = 20)
    #sps[1,5].text(-0.05, -0.2 , f'p = {np.round(pvals_all_l_hc[0], 3)}', fontweight = 'bold', fontsize = 14)
    sps[1,5].text(-0.05, -0.2 , f't = {np.round(sum(t_l_hc[np.where(mask_l_hc[:, 2] ==1)[0], 2]), 2)}', fontweight = 'bold', fontsize = 14)
    
    
    
    
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.01, pad=0.04)
    cb.set_label('t-value', rotation = 90, fontsize = 12)
    plt.show()
    fig.savefig(save_folder_peak + 'N3_cluster_bilaterality' + '.svg') 
    
    

 
def bar_plot_n2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_n2 = {'Control': lat_t_hc[component_number, :], 'Stroke V2': np.concatenate((lat_r_arr_st[component_number, :, 0], lat_l_arr_st[component_number, :, 0])),
                                          'Stroke V3': np.concatenate((lat_r_arr_st[component_number, :, 1], lat_l_arr_st[component_number, :, 1])),
                                          'Stroke V4': np.concatenate((lat_r_arr_st[component_number, :, 2], lat_l_arr_st[component_number, :, 2])),
                                          'Stroke V5': np.concatenate((lat_r_arr_st[component_number, :, 3], lat_l_arr_st[component_number, :, 3])), 
                                          'Stroke V6': np.concatenate((lat_r_arr_st[component_number, :, 4], lat_l_arr_st[component_number, :, 4]))}
    fig, ax = plt.subplots()
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    data_mean = [np.mean(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    ax.plot(np.array([1, 2, 3, 4, 5]), data_mean[1:], color = 'k')
    ax.scatter(.45, 98, marker = '*', color = 'k')
    ax.plot([0, 1], [90, 90], color = 'k')
    ax.plot([0, 0, 0, 0], [90, 89, 88, 87], color = 'k')
    ax.plot([1, 1, 1, 1], [90, 89, 88, 87], color = 'k')
    ax.scatter(3, 109, marker = '*', color = 'k')
    ax.plot([1, 2, 3, 4, 5], [100, 100, 100, 100, 100], color = 'k')
    ax.plot([1, 1, 1, 1], [100, 99, 98, 97], color = 'k')
    ax.plot([5, 5, 5, 5], [100, 99, 98, 97], color = 'k')
    ax.bar( x = np.arange(len(dict_n2.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['C','C', 'S-V2', 'S-V3', 'S-V4', 'S-V5', 'S-V6'], fontsize=12) 
    ax.set_ylabel('Latency (ms)', fontsize = 14)
    ax.set_title('N60 Latency', fontweight="bold")
    ax.set_ylim([0, 130])
    t_ind , p_ind = stats.ttest_ind(dict_n2[str('Control')], dict_n2[str('Stroke V2')])
    t_paired, p_paired = stats.ttest_rel(dict_n2[str('Stroke V2')], dict_n2[str('Stroke V6')])
    ax.text(-0.3 , 120, f'P = {np.round(p_ind, 3)}', fontweight = 'bold', fontsize = 12)
    ax.text(2 , 120, f'P = {np.round(p_paired, 3)}', fontweight = 'bold', fontsize = 12)
    plt.show()
    fig.savefig(save_folder_peak  + 'n2_latency_barplot'+ '.svg', overwrite = True)
    

    
    return(t_ind , p_ind, t_paired, p_paired, fig)


    
    
def bar_plot_p2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak):

    dict_n2 = {'Control': lat_t_hc[component_number, :], 'Stroke V2': np.concatenate((lat_r_arr_st[component_number, :, 0], lat_l_arr_st[component_number, :, 0])),
                                          'Stroke V3': np.concatenate((lat_r_arr_st[component_number, :, 1], lat_l_arr_st[component_number, :, 1])),
                                          'Stroke V4': np.concatenate((lat_r_arr_st[component_number, :, 2], lat_l_arr_st[component_number, :, 2])),
                                          'Stroke V5': np.concatenate((lat_r_arr_st[component_number, :, 3], lat_l_arr_st[component_number, :, 3])), 
                                          'Stroke V6': np.concatenate((lat_r_arr_st[component_number, :, 4], lat_l_arr_st[component_number, :, 4]))}
    fig, ax = plt.subplots()
    data_std = np.zeros([1, 6])
    data_mean =  np.zeros([1, 6])
    data_std = [np.std(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    data_mean = [np.mean(dict_n2[str(j)]) for i,j in enumerate(list(dict_n2.keys()))]
    ax.plot(np.array([1, 2, 3, 4, 5]), data_mean[1:], color = 'k')
    ax.plot([0, 1], [220, 220], color = 'k')
    ax.plot([0, 0, 0, 0], [219, 218, 217, 216], color = 'k')
    ax.plot([1, 1, 1, 1], [219, 218, 217, 216], color = 'k')
    ax.plot([1, 2, 3, 4, 5], [230, 230, 230, 230, 230], color = 'k')
    ax.plot([1, 1, 1, 1], [229, 228, 227, 226], color = 'k')
    ax.plot([5, 5, 5, 5], [229, 228, 227, 226], color = 'k')
    ax.bar( x = np.arange(len(dict_n2.keys())), height = data_mean, yerr= data_std, capsize=4, color ='grey', alpha = 0.7)
    ax.set_xticklabels(['C','C', 'S-V2', 'S-V3', 'S-V4', 'S-V5', 'S-V6'], fontsize=12) 
    ax.set_ylabel('Latency (ms)', fontsize = 14)
    ax.set_title('P190 Latency', fontweight="bold")
    ax.set_ylim([0, 260])
    t_ind , p_ind = stats.ttest_ind(dict_n2[str('Control')], dict_n2[str('Stroke V2')])
    t_paired, p_paired = stats.ttest_rel(dict_n2[str('Stroke V2')], dict_n2[str('Stroke V6')])
    ax.text(-0.3 , 240, f'P = {np.round(p_ind, 3)}', fontweight = 'bold', fontsize=12)
    ax.text(2 , 240, f'P = {np.round(p_paired, 3)}', fontweight = 'bold', fontsize=12)
    plt.show()
    fig.savefig(save_folder_peak  + 'n120_latency_barplot'+ '.svg', overwrite = True)
    

    
    return(t_ind , p_ind, t_paired, p_paired, fig)
   
    
def FM_UE_plotting(amp_l_arr_st, amp_r_arr_st, fuma_score_swap, save_folder_peak): 

    n_component = 4  
    fig, ax  = plt.subplots(1, 2, figsize= (11, 4))
    ax[0].set_xlabel('FMUE', fontweight ='bold', fontsize = 12)
    ax[0].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 0]
    Y = np.concatenate((amp_l_arr_st[n_component,:, 0], amp_r_arr_st[n_component,:, 0]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[0].scatter(X, Y, color = 'k')
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[0].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[0].text(14, 1., f'R = {np.round(r, 3)}, p = {np.round(p, 4)}', fontweight = 'bold', fontsize = 12)
    ax[0].set_xlim(5, 25)
    ax[0].set_ylim(-1, 2)
    ax[0].set_title('V2: FMUE vs. Amplitude of \n Ipsi Sensorimotor Region', fontweight = 'bold' , fontsize = 12)
    ax[0].set_title('A', loc='left', y=1.07, x=0.05, fontweight = 'bold',  fontsize = 14)

    
    
    ax[1].set_xlabel('FMUE', fontweight ='bold', fontsize = 12)
    ax[1].set_ylabel('Amplitude', fontweight ='bold', fontsize = 12)
    X = fuma_score_swap[:, 4]
    Y = np.concatenate((amp_l_arr_st[n_component,:, 4], amp_r_arr_st[n_component,:, 4]))
    r, p = scipy.stats.pearsonr(X, Y)
    ax[1].scatter(X, Y, color = 'k')
    model = LinearRegression()
    X = X.reshape(-1, 1)
    model.fit(X, Y)
    ax[1].plot(X, model.predict(X), color='red', label='Regression Line')
    ax[1].text(14, 1., f'R = {np.round(r, 3)}, p = {np.round(p, 4)}', fontweight = 'bold', fontsize = 12)
    ax[1].set_xlim(5, 25)
    ax[1].set_ylim(-1, 2)
    ax[1].set_title('V6: FMUE vs. Amplitude of \n Ipsi Sensorimotor Region', fontweight = 'bold' , fontsize = 12)
    ax[1].set_title('B', loc='left', y = 1.07, x=0.01, fontweight = 'bold',  fontsize = 14)

    plt.show()
    fig.savefig(save_folder_peak + 'FM_UE_sensorimotor.tif', dpi = 600, pil_kwargs={'compression': 'tiff_lzw'} , overwrite = True)

 
def contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component, text):    
    data_contra_s = np.zeros([10, end_time-1000])
    evokeds_all_L_corrected = evokeds_all_L[str('v2')]
    for i1,i in enumerate(evokeds_all_R[str('v2')]):
        data_contra_s[i1+6, :] = np.mean(evokeds_all_R[str('v2')][i1].data[contra_right_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L[str('v2')]):
        data_contra_s[i1, :] = np.mean(evokeds_all_L_corrected[i1].data[contra_left_ind,  1000:end_time], axis = 0)
        
    data_ipsi_s = np.zeros([10, end_time-1000])
    for i1,i in enumerate(evokeds_all_R[str('v2')]):
        data_ipsi_s[i1+6, :] = np.mean(evokeds_all_R[str('v2')][i1].data[contra_left_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L[str('v2')]):
        data_ipsi_s[i1, :] = np.mean(evokeds_all_L_corrected[i1].data[contra_right_ind,  1000:end_time], axis = 0)
        
        
           
        
    data_contra_c = np.zeros([17, end_time-1000])
    for i1,i in enumerate(evokeds_all_R_hc):
        data_contra_c[i1+11, :] = np.mean(evokeds_all_R_hc[i1].data[contra_right_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L_hc):
        data_contra_c[i1, :] = np.mean(evokeds_all_L_hc[i1].data[contra_left_ind,  1000:end_time], axis = 0)
        
        
    data_ipsi_c = np.zeros([17, end_time-1000])
    for i1,i in enumerate(evokeds_all_R_hc):
        data_ipsi_c[i1+11, :] = np.mean(evokeds_all_R_hc[i1].data[contra_left_ind,  1000:end_time], axis = 0)    
    for i1,i in enumerate(evokeds_all_L_hc):
        data_ipsi_c[i1, :] = np.mean(evokeds_all_L_hc[i1].data[contra_right_ind,  1000:end_time], axis = 0)
    
    
    data_ipsi_s[1, 0:60]= np.mean(data_ipsi_s[1, 60:65])
    
    data_ipsi_c[14, 0:40]= np.mean(data_ipsi_c[14, 60:65])
    data_ipsi_c[15, 0:60]= np.mean(data_ipsi_c[15, 60:65])
    def confidence_interval(data):
        # Parameters
        confidence_level = 0.95  # 95% confidence interval
        alpha = 1 - confidence_level
        
        # Calculate the mean and standard error for each time point across all time series
        means = np.mean(data, axis=0)
        sems = stats.sem(data, axis=0)
        
        # Calculate the margin of error
        margin_of_error = sems * stats.t.ppf(1 - alpha/2., data.shape[0] - 1)
        
        # Calculate the confidence interval
        lower_bound = means - margin_of_error
        upper_bound = means + margin_of_error
        
        return(lower_bound, upper_bound, means, sems)
    
    # Plotting
    fig1 = plt.figure()
    time_points = evokeds_all_R[str('v2')][0].times[1000:end_time]
    # stroke
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_contra_s, axis  =1))
    plt.plot(time_points, means, label='S-V2', color='r')
    plt.fill_between(time_points, means - sems, means + sems, color='r', alpha=0.1)
    # control
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_contra_c, axis  =1 ))
    plt.plot(time_points, means, label='C', color='k')
    plt.fill_between(time_points, means - sems, means + sems, color='k', alpha=0.15)
    plt.xlabel('Latency (ms)', fontsize = 14)
    plt.ylim(-2, 2)
    plt.ylabel('Mean Amplitude (µV)', fontsize = 14)
    if component == 'P190':
        plt.title(f'{component} response', fontweight = 'bold')
    else:
    
        plt.title(f'{component} Contralateral response', fontweight = 'bold')
    if component == 'P30':
        plt.vlines(x = [0.02, 0.05], ymin = -1, ymax = 1, color = 'b', linewidth = 1)
        plt.vlines(x = [0.1, 0.135], ymin = -1, ymax = 1, color = 'b', linewidth = 1)
        plt.text(0.03, 1.6, 'P30', color ='b', fontweight= 'bold', size =12)
        plt.text(0.105, -1.5, 'N120', color ='b', fontweight= 'bold', size =12)
    elif component == 'N60':
         plt.vlines(x = [0.05, 0.08], ymin = -1.8, ymax = 0, color = 'b', linewidth = 1)
         plt.text(0.055, 0.3, 'N60', color ='b', fontweight= 'bold', size =12)
    elif component == 'P190':
         plt.vlines(x = [0.14, 0.22], ymin = -1, ymax = 1, color = 'b', linewidth = 1)
         plt.text(0.17, 1.3, 'P190', color ='b', fontweight= 'bold', size =12)
        
# =============================================================================
#     if component == 'N2':
#         ax[0].vlines(x = [0.015, 0.05], ymin = 0, ymax = 1.5, color = 'grey', linewidth = 1, linestyle = '-.')
#     elif component == 'N1':
#         ax[0].vlines(x = [0.045, 0.080], ymin = 0, ymax = -1.8, color = 'grey', linewidth = 1, linestyle = '-.')
#     elif component == 'P2':
#         ax[0].vlines(x = [0.140, 0.22], ymin = -0.5, ymax = 1, color = 'grey', linewidth = 1, linestyle = '-.')
# =============================================================================
        
    plt.legend()
    plt.show()
    
    fig2 = plt.figure()
    
    # Plotting
    time_points = evokeds_all_R[str('v2')][0].times[1000:end_time]
    # stroke
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_ipsi_s, axis = 1))
    plt.plot(time_points, means, label='S-V2', color='r')
    plt.fill_between(time_points, means - sems, means + sems, color='r', alpha=0.2)
    # control
    lower_bound, upper_bound, means, sems = confidence_interval(zscore(data_ipsi_c, axis =1))
    plt.plot(time_points, means, label='C', color='k')
    plt.fill_between(time_points, means - sems, means + sems, color='k', alpha=0.2)
    plt.xlabel('Latency (ms)', fontsize = 14)
    plt.ylim(-2, 2)
    plt.ylabel('Mean Amplitude (µV)', fontsize = 14)
    if component == 'P190':
        plt.title(f'{component} response', fontweight = 'bold')
    else:
        
        plt.title(f'{component} Ipsilateral response', fontweight = 'bold')
    if component == 'N120':
        plt.vlines(x = [0.100, 0.135], ymin = -1, ymax = 1, color = 'b', linewidth = 1)
        plt.text(0.105, -1.5, 'N120', color ='b', fontweight= 'bold', size =12)
# =============================================================================
#         plt.vlines(x = [0.100, 0.135], ymin = -1, ymax = 1, color = 'b', linewidth = 1)
#         plt.text(0.105, -1.5, 'P120', color ='b', fontweight= 'bold', size =12)
# =============================================================================
    plt.legend()
    plt.show()
    #fig1.suptitle(f'{text}')
    #fig2.suptitle(f'{text}')
    fig1.savefig(save_folder_peak  + f'{text}'+ 'contra_ipsi_william'+ '.svg', overwrite = True)
    fig2.savefig(save_folder_peak  + f'{text}'+ 'contra_ipsi_william'+ '.svg', overwrite = True)
    return fig1, fig2
    

    
def amp_latency_6_components_hc(evokeds_all_L_hc, evokeds_all_R_hc, contra_right, contra_left, time_windows, save_folder_peak):    
    amp_r = {}
    lat_r = {}
    amp_l = {}
    lat_l = {}
    amp_lat_r = {}
    amp_lat_l = {}
    
   
    for i_r,_ in enumerate(range(len(evokeds_all_R_hc))):
        amp_lat_r[str(i_r)] = {}
        amp_r[str(i_r)] = np.zeros(len(time_windows))
        lat_r[str(i_r)] = np.zeros(len(time_windows))
        
        for i_contra_right, n_contra_right in enumerate(contra_right):  
            amp_lat_r[str(i_r)][str(n_contra_right)] = np.zeros([len(contra_right), 1001])
            contra_right_ind = channel_indices(contra_right[str(n_contra_right)])
            a = zscore(evokeds_all_R_hc[i_r].data[:, ], axis =0)
            a = (evokeds_all_R_hc[i_r].data[:, ])
            amp_lat_r[str(i_r)][str(n_contra_right)] = np.mean(a[contra_right_ind, 1000:], axis = 0) 
            
            
            if i_contra_right == 0:
                color = 'g'
                amp_r[str(i_r)][i_contra_right] = np.max(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

                
            elif i_contra_right == 1:
                color = 'g'
                amp_r[str(i_r)][i_contra_right] = np.min(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

            elif i_contra_right == 2:
                color = 'navy'
                amp_r[str(i_r)][i_contra_right] = np.min(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmin(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

                
            elif i_contra_right == 3:
                color = 'maroon'
                amp_r[str(i_r)][i_contra_right] = np.max(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]

            elif i_contra_right == 4:
                color = 'royalblue'
                amp_r[str(i_r)][i_contra_right] = np.max(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]])
                lat_r[str(i_r)][i_contra_right] = np.argmax(amp_lat_r[str(i_r)][str(n_contra_right)][time_windows[n_contra_right][0]: time_windows[n_contra_right][1]]) + time_windows[n_contra_right][0]


    
    
    
    #############################
    # Left Side
    
    for i_l,_ in enumerate(range(len(evokeds_all_L_hc))):
        amp_lat_l[str(i_l)] = {}
        amp_l[str(i_l)] = np.zeros(len(time_windows))
        lat_l[str(i_l)] = np.zeros(len(time_windows))

            
        for i_contra_left, n_contra_left in enumerate(contra_left):  
            amp_lat_l[str(i_l)][str(n_contra_left)] = np.zeros([len(contra_left), 1001])
            contra_left_ind = channel_indices(contra_left[str(n_contra_left)])
            a = zscore(evokeds_all_L_hc[i_l].data[:, ], axis =0)
            amp_lat_l[str(i_l)][str(n_contra_left)] = np.mean(evokeds_all_L_hc[i_l].data[contra_left_ind, 1000:], axis = 0) 
    
            

            
            if i_contra_left == 0:
                color = 'g'
                amp_l[str(i_l)][i_contra_left] = np.min(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

                
            elif i_contra_left == 1:
                color = 'g'
                amp_l[str(i_l)][i_contra_left] = np.max(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

            elif i_contra_left == 2:
                color = 'navy'
                amp_l[str(i_l)][i_contra_left] = np.min(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

                
            elif i_contra_left == 3:
                color = 'maroon'
                amp_l[str(i_l)][i_contra_left] = np.max(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmax(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

            elif i_contra_left == 4:
                color = 'royalblue'
                amp_l[str(i_l)][i_contra_left] = np.min(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]])
                lat_l[str(i_l)][i_contra_left] = np.argmin(amp_lat_l[str(i_l)][str(n_contra_left)][time_windows[n_contra_left][0]: time_windows[n_contra_left][1]]) + time_windows[n_contra_left][0]

                

    
    
    ### 
    
    
    amp_l_arr_hc = np.zeros([5, 11])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            amp_l_arr_hc[i_components,i] = amp_l[str(i)][i_components]
    
    
    lat_l_arr_hc = np.zeros([5, 11])
    for i,_ in enumerate(amp_l):
        for i_components, n_components in enumerate(time_windows): 
            lat_l_arr_hc[i_components,i] = lat_l[str(i)][i_components]
    
    
    amp_r_arr_hc = np.zeros([5, 6])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            amp_r_arr_hc[i_components,i] = amp_r[str(i)][i_components]
            
    lat_r_arr_hc = np.zeros([5, 6])
    for i,_ in enumerate(amp_r):
        for i_components, n_components in enumerate(time_windows): 
            lat_r_arr_hc[i_components,i] = lat_r[str(i)][i_components]
    
    
    
    
    amp_t = np.concatenate((amp_l_arr_hc, amp_r_arr_hc), axis = 1)
    lat_t = np.concatenate((lat_l_arr_hc, lat_r_arr_hc), axis = 1)
    


            
    return(amp_t, lat_t)        





def mark_bad_channels_interpolate(raw):

    """
    Detects channels above a certain threshold when looking at zscored data.
    Plots time series with pre-detected channels marked in red.
    Enables user to mark bad channels interactively and saves selection in raw object.

    Args:
        raw : MNE raw object with EEG data
        ch_names (list): list of strings with channel names
        threshold (float, int): threshold based on which to detect outlier channels 
                                (maximal zscored absolute standard deviation). Defaults to 1.5.

    Returns:
        MNE raw object: raw, with bad channel selection (bads) updated
    """


    ch_names = raw.info['ch_names']


    # plotting of channel variance
    vars = np.var(raw._data.T, axis=0)
    badchans_threshold = np.where(np.abs(zscore(vars)) > 1.5)
    #badchans = visual_inspection(vars)
    raw.info['bads'] = [ch_names[i] for i in list(badchans_threshold[0])]

    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)
    ch_names = raw.info['ch_names']
    badchans_threshold  = raw.info['bads']
    raw_eeg_interp = raw.interpolate_bads(reset_bads=True)
    
    

    return raw_eeg_interp, badchans_threshold


def clean_dataset(epochs):
    """Create cleaned dataset (by running autoreject and ICA)
    with each subject data in a dictionary.
    Parameter
    ----------
    subject : string of subject ID e.g. 7707
    trial   : HighFine, HighGross, LowFine, LowGross
    Returns
    ----------
    clean_eeg_dataset : dataset of all the subjects with different conditions
    """
    data  = {}
    ica_epochs, ica = clean_with_ica(epochs)
    repaired_eeg = autoreject_repair_epochs(ica_epochs)
    data['eeg'] = repaired_eeg
    data['ica'] = ica


    return data
        