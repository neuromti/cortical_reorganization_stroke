
"""
Created on Tue May 26 08:21:24 2026

@author: sara
"""

import pickle
import mne
import scipy.stats
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import meta_functions_HC as function_hc
from mne.channels import make_standard_montage
from scipy.signal import find_peaks
import statistics

from scipy.signal import medfilt



#%%
evokeds_pahse_effecft_all = []
epochs_data_dir = "/home/sara/data/Third part/epochs/"
epochs_files = Path(epochs_data_dir).glob('*epo.fif*')
plt.close('all')
Save_folder = "/home/sara/data/Third part/epochs/V2_V6/"
# Below I load the patient data, I have removed the names for privacy reasons. It would look like the name and the stimulated hand

dictionary_bad_channels = { 
                                               
}


stim_artifact_subs = {
                      }



evokeds_all_L = {}
evokeds_all_R = {}
sub_names_L = {}
sub_names_R = {}
for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
    evokeds_all_R[str(time_point)] = []
    evokeds_all_L[str(time_point)] = []
    sub_names_L[str(time_point)] = []
    sub_names_R[str(time_point)] = []


for f in epochs_files:
    plt.close('all')
    subject_ID = f.parts[-1][0:-8]
    print(f.parts[-1])
    

    if subject_ID in dictionary_bad_channels:
        epochs = mne.read_epochs(f, preload= True)
        epochs = epochs.set_eeg_reference(ref_channels='average')      
        evokeds = epochs.average()     
        montage = make_standard_montage('standard_1005')
        epochs = epochs.set_montage(montage)
        
        
        
        all_times = np.arange(0, 0.5, 0.01)
        topo_plots = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11))
        ERP_plots = evokeds.plot(spatial_colors = True, gfp = True) 
        ERP_plots.set_size_inches((20, 8))
       
        epochs.info['bads'] = dictionary_bad_channels[subject_ID]
        epochs_clean = epochs.interpolate_bads(reset_bads=True, mode='accurate')

        
        # cubic interpolation for some subjects that have sth like a TMS artifact
        for sub_name, sub_name_v in enumerate(stim_artifact_subs): 
            if (subject_ID == sub_name_v):
                print(sub_name_v)
                epochs_clean = function_hc.cubic_interp(epochs_clean, win = stim_artifact_subs[sub_name_v])
                epochs_clean = epochs_clean
    
        evokeds_clean = epochs_clean.average()    
        all_times = np.arange(0, 0.5, 0.01)
        topo_plots = evokeds_clean.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11))
        ERP_plots = evokeds_clean.plot(spatial_colors = True, gfp = True) 
        ERP_plots.set_size_inches((20, 8))
        topo_plots_senors = evokeds_clean.plot_topomap(np.arange(0, 0.4, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
        topo_plots_senors.savefig(Save_folder+ 'figs/each_sub/' + f'{subject_ID}' + '_topo'    + '.svg', overwrite = True) 
        ERP_plots.savefig(Save_folder+ 'figs/each_sub/' + f'{subject_ID}' + '_erp'    + '.svg', overwrite = True) 
    
        

    
        if subject_ID[-1] == 'R':
            if subject_ID[5:9] == 'pre1' or  subject_ID[5:9] == 'pre2':
                evokeds_all_R[str('v2')].append(evokeds_clean)
                sub_names_R[str('v2')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v2/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:9] == 'pre3' or  subject_ID[5:9] == 'pre4':
                evokeds_all_R[str('v3')].append(evokeds_clean)
                sub_names_R[str('v3')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v3/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:10] == 'post1' or  subject_ID[5:10] == 'post2':
                evokeds_all_R[str('v4')].append(evokeds_clean)
                sub_names_R[str('v4')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v4/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:10] == 'post3' or  subject_ID[5:10] == 'post4':
                evokeds_all_R[str('v5')].append(evokeds_clean)
                sub_names_R[str('v5')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v5/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:10] == 'post5' or  subject_ID[5:10] == 'post6':
                evokeds_all_R[str('v6')].append(evokeds_clean)
                sub_names_R[str('v6')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Right/'  + '/v6/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
  
        elif subject_ID[-1] == 'L':
            if subject_ID[5:9] == 'pre1' or  subject_ID[5:9] == 'pre2':
                evokeds_all_L[str('v2')].append(evokeds_clean)
                sub_names_L[str('v2')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v2/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
                
            elif subject_ID[5:9] == 'pre3' or  subject_ID[5:9] == 'pre4':
                evokeds_all_L[str('v3')].append(evokeds_clean)
                sub_names_L[str('v3')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v3/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')

                
            elif subject_ID[5:10] == 'post1' or  subject_ID[5:10] == 'post2':
                evokeds_all_L[str('v4')].append(evokeds_clean)
                sub_names_L[str('v4')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v4/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')

                    
            elif subject_ID[5:10] == 'post3' or  subject_ID[5:10] == 'post4':
                evokeds_all_L[str('v5')].append(evokeds_clean)
                sub_names_L[str('v5')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v5/'  + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')

                
            elif subject_ID[5:10] == 'post5' or  subject_ID[5:10] == 'post6':
                evokeds_all_L[str('v6')].append(evokeds_clean)
                sub_names_L[str('v6')].append(subject_ID[0:4])
                epochs_clean.save(Save_folder + '/Left/'  + '/v6/' + str(f.parts[-1][0:-8]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
              
                
             
Evoked_GrandAv_R = {} 
Evoked_GrandAv_L = {}           
for _,time_point in enumerate(['v2', 'v3', 'v4', 'v5', 'v6']):
    Evoked_GrandAv_R[str(time_point)] =  mne.grand_average(evokeds_all_R[str(time_point)]) 
    Evoked_GrandAv_L[str(time_point)] = mne.grand_average(evokeds_all_L[str(time_point)])             
             
             

    # Right side plots
    ERP_plots_R = Evoked_GrandAv_R[str(time_point)].plot(spatial_colors = True, gfp = True) 
    ERP_plots_R.set_size_inches((20, 8))
    Evoked_GrandAv_R_fig = Evoked_GrandAv_R[str(time_point)].crop(-0.015, 0.5).plot_joint(times= [0.040, 0.060, 0.120, 0.150], ts_args = dict( ylim=dict(eeg=[-2, 2]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-2, 2], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
    topo_plots_senors_r = Evoked_GrandAv_R[str(time_point)].plot_topomap(np.arange(0, 0.4, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
    


    # Left Side plots
    ERP_plots_L = Evoked_GrandAv_L[str(time_point)].plot(spatial_colors = True, gfp = True) 
    ERP_plots_L.set_size_inches((20, 8))
    topo_plots_senors_L = Evoked_GrandAv_L[str(time_point)].plot_topomap(np.arange(0, 0.4, 0.01), ch_type='eeg', time_unit='s', ncols=8, nrows='auto',  sphere=(0.00, 0.00, 0.00, 0.11), scalings = dict(eeg=1), vlim=(-2,2))
    Evoked_GrandAv_L_fig = Evoked_GrandAv_L[str(time_point)].crop(-0.015, 0.5).plot_joint(times=[0.040, 0.060, 0.120, 0.150], ts_args = dict( ylim=dict(eeg=[-2, 2]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-2, 2], units = dict(eeg='µV')  , sphere=(0.00, 0.00, 0.00, 0.11)))
    

    # Save Figures
    topo_plots_senors_r.savefig(Save_folder + 'figs/' +'R_' + str(time_point) + '_topo_plots_senors_patients_'  + '.svg', overwrite = True) 
    Evoked_GrandAv_R_fig.savefig(Save_folder+ 'figs/' +'R_' + str(time_point) + '_Evoked_GrandAv_patients_'     + '.svg', overwrite = True) 
    topo_plots_senors_L.savefig(Save_folder + 'figs/' +'L_' + str(time_point) + '_topo_plots_senors_patients_'  + '.svg', overwrite = True) 
    Evoked_GrandAv_L_fig.savefig(Save_folder+ 'figs/' +'L_' + str(time_point) + '_Evoked_GrandAv_patients_'     + '.svg', overwrite = True) 








save_folder_peak = '/home/sara/data/Third part/epochs_manually_rejected/V2/peak_amp_compare/'
# saving evoked files
mne.evoked.write_evokeds(save_folder_peak + 'ST_V2_L_ave.fif', Evoked_GrandAv_L[str('v2')], overwrite = True)
mne.evoked.write_evokeds(save_folder_peak + 'ST_V2_R_ave.fif', Evoked_GrandAv_R[str('v2')], overwrite = True)






#%%

with open(str(save_folder_peak) + 'ST_ERP_R_V2_V6.p', 'wb') as fp:
    pickle.dump(Evoked_GrandAv_R, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open(str(save_folder_peak) + 'ST_ERP_L_V2_V6.p', 'wb') as fp:
    pickle.dump(Evoked_GrandAv_L, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    


#############
#%%
Evoked_GrandAv_L_v2 = mne.grand_average(evokeds_all_L[str('v2')])
Evoked_GrandAv_L_v2.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_L_v2.info['bads'] = ['FT10', 'FT8', 'F8', 'TP8', 'P8', 'T8', 'PO8', 'FT10', 'Fpz', 'TP9', 'O1', 'PO7', 'TP7', 'P7', 'T7']
Evoked_GrandAv_L_v2 = Evoked_GrandAv_L_v2.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_v2_plot_l = Evoked_GrandAv_L_v2.plot_joint(times = [0.04, 0.07, 0.12, 0.190],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), units = dict(eeg="Amplitude"), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='AmpµV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_v2_plot_l.set_size_inches((7, 6))
grand_average_v2_plot_l.axes[0].set_ylabel('Amplitude (µV)') 
grand_average_v2_plot_l.savefig(save_folder_peak  + 'grand_average_v2_plot_l.png', overwrite = True)



Evoked_GrandAv_R_v2 = mne.grand_average(evokeds_all_R[str('v2')])
Evoked_GrandAv_R_v2.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_R_v2.info['bads'] = ['TP9', 'PO7', 'O1', 'Oz']
Evoked_GrandAv_R_v2 = Evoked_GrandAv_R_v2.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_v2_plot_r = Evoked_GrandAv_R_v2.plot_joint(times = [0.04, 0.07, 0.12, 0.190],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_v2_plot_r.set_size_inches((7, 6))
grand_average_v2_plot_r.axes[0].set_ylabel('Amplitude (µV)')
grand_average_v2_plot_r.savefig(save_folder_peak  + 'grand_average_v2_plot_r.png', overwrite = True)

#%%

import meta_functions_HC as function_hc

with open(str(save_folder_peak) + 'evokeds_all_L_hc.p', 'rb') as fp:
    evokeds_all_L_hc = pickle.load(fp)

with open(str(save_folder_peak) + 'evokeds_all_R_hc.p', 'rb') as fp:
    evokeds_all_R_hc = pickle.load(fp)
    



Evoked_GrandAv_L_hc = mne.grand_average(evokeds_all_L_hc)
Evoked_GrandAv_L_hc.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_L_hc.info['bads'] = ['FT10', 'FT8', 'F8', 'TP8', 'P8', 'T8', 'PO8', 'FT10', 'Fpz', 'TP9', 'O1', 'PO7', 'TP7', 'P7', 'T7']
Evoked_GrandAv_L_hc = Evoked_GrandAv_L_hc.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_hc_plot_l = Evoked_GrandAv_L_hc.plot_joint(times = [0.03, 0.06, 0.12, 0.19],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_hc_plot_l.set_size_inches((7, 6))
grand_average_hc_plot_l.axes[0].set_ylabel('Amplitude (µV)')
grand_average_hc_plot_l.savefig(save_folder_peak  + 'Evoked_GrandAv_L_hc_plot.png', overwrite = True)



Evoked_GrandAv_R_hc = mne.grand_average(evokeds_all_R_hc)
Evoked_GrandAv_R_hc.crop(tmin = -0.06, tmax=0.5)
Evoked_GrandAv_R_hc.info['bads'] = ['TP9', 'PO7', 'O1', 'Oz']
Evoked_GrandAv_R_hc = Evoked_GrandAv_R_hc.interpolate_bads(reset_bads=True, mode='accurate')
grand_average_hc_plot_r = Evoked_GrandAv_R_hc.plot_joint(times = [0.03, 0.06, 0.12, 0.190],  ts_args = dict( ylim=dict(eeg=[-2.5, 2.5]), scalings = dict(eeg=1)), topomap_args = dict(scalings = dict(eeg=1), vlim=[-1.5, 1.5], units = dict(eeg='µV') , sphere=(0.00, 0.00, 0.00, 0.11)))
grand_average_hc_plot_r.set_size_inches((7, 6))
grand_average_hc_plot_r.axes[0].set_ylabel('Amplitude (µV)')
grand_average_hc_plot_r.savefig(save_folder_peak  + 'Evoked_GrandAv_R_hc_plot.png', overwrite = True)

#%%








import meta_functions_HC as function_hc



save_folder =  '/home/sara/data/Third part/epochs_manually_rejected/V2_V6/'
exdir_epoch_r = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Right/"
exdir_epoch_l = "/home/sara/data/Third part/epochs_manually_rejected/V2_V6/Left/"


plt.close('all')
win_erp0 = [35, 50]   
win_erp1 = [65, 100]   
win_erp2 = [100, 135] 
win_erp3 = [140, 220] 



labels = ['P1', 'N1', 'N2', 'P2']
win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])
ch_names_r = {}; ch_names_l = {}; pvals_all_r = {}; pvals_all_l = {}; peaks_r  = {}; peaks_l  = {}
mean_peaks_r  = {}; mean_peaks_l  = {}; mask_r = {}; mask_l = {}; t_l = {}; t_r = {}; num_sub_l = 6; num_sub_r = 4


time_points = ['v2', 'v3', 'v4', 'v5', 'v6'] 




for _,time_point in enumerate(time_points):
    ch_names_r[str(time_point)], pvals_all_r[str(time_point)], t_r[str(time_point)], mask_r[str(time_point)], pos, peaks_r[str(time_point)]  = function_hc.clustering_channels(4, win_erps, exdir_epoch_r + f'{time_point}/',  [0.5, 2.5, 2, 2], labels, 'R_'+ f'{time_point}', save_folder + 'Right/')    
    #win_erp0 = [35, 50]   ;win_erp1 = [65, 100]   ;win_erp2 = [100, 135] ;win_erp3 = [170, 200] ;win_erps = np.array([win_erp0, win_erp1, win_erp2, win_erp3])

    ch_names_l[str(time_point)], pvals_all_l[str(time_point)], t_l[str(time_point)], mask_l[str(time_point)], pos, peaks_l[str(time_point)]  = function_hc.clustering_channels(6, win_erps, exdir_epoch_l + f'{time_point}/',  [1.5, 0.5, 2, 0.8], labels, 'L_'+ f'{time_point}', save_folder + 'Left/')    

    
    




    
#%%

import meta_functions_HC as function_hc
contra_right = ['C1', 'C3', 'CP1', 'CP3', 'C5', 'CP5']
contra_left =  ['C2', 'C4', 'CP2', 'CP4', 'C6', 'CP6']


fig_LI_in = function_hc.laterality_Index_errorbar(contra_right, contra_left, evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, save_folder_peak)

#%%
# real clusters
import meta_functions_HC as function_hc

save_folder =  '/home/sara/data/Second part/epochs_manually_rejected/Group_models/'
exdir_epoch_r = "/home/sara/data/Second part/epochs_manually_rejected/Right/"
exdir_epoch_l = "/home/sara/data/Second part/epochs_manually_rejected/Left/"

save_folder_peak = '/home/sara/data/Third part/epochs_manually_rejected/V2/peak_amp_compare/'


win_erp0 = [10, 35];  win_erp1 = [60, 90];   win_erp2 = [100, 135];  win_erp3 = [140, 180]; win_erps  = np.array([win_erp0, win_erp1, win_erp2, win_erp3])



ch_names_r_hc, pvals_all_r_hc, t_r_hc, mask_r_hc, pos, peaks  = function_hc.clustering_channels(6, win_erps, exdir_epoch_r,  [1.4, 2, 2, 0.8], labels, 'R', save_folder + '/right/')    



ch_names_l_hc, pvals_all_l_hc, t_l_hc, mask_l_hc, pos, peaks  = function_hc.clustering_channels(11, win_erps, exdir_epoch_l, [2, 2.5, 0.8, 0.5], labels, 'L',save_folder + '/left/')    


#%%


function_hc.N3_cluster_bilaterality(t_r, t_l, pvals_all_r, pvals_all_l, mask_r, mask_l, pos, t_r_hc, t_l_hc, pvals_all_r_hc, pvals_all_l_hc, mask_r_hc, mask_l_hc, save_folder_peak)


#%%



#% P1 N2 contra_ipsi_william
import meta_functions_HC as function_hc

end_time = 1270
contra_right = ['C1', 'C3', 'CP1', 'CP3', 'C5', 'CP5']
contra_left =  ['C2', 'C4', 'CP2', 'CP4', 'C6', 'CP6']
contra_right_ind = function_hc.channel_indices(contra_right)
contra_left_ind = function_hc.channel_indices(contra_left)
contra_p30, ipsi_n120 = function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'P30', text = 'P30')
contra_N120, ipsi_p120  = function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'N120', text = 'N120')



#% N1 contra_ipsi_william
contra_right = ['F1', 'F3', 'FC1', 'FC3']
contra_left =  ['F2', 'F4', 'FC2', 'FC4']
contra_right_ind = function_hc.channel_indices(contra_right)
contra_left_ind = function_hc.channel_indices(contra_left)
contra_n60, ipsi_N60 = function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'N60', text = 'N60')





#% P2 contra_ipsi_william
contra_right = ['Cz']
contra_left =  ['Cz']
contra_right_ind = function_hc.channel_indices(contra_right)
contra_left_ind = function_hc.channel_indices(contra_left)
contra_190, ipsi_p190 = function_hc.contra_ipsi_william(evokeds_all_L, evokeds_all_R, evokeds_all_L_hc, evokeds_all_R_hc, contra_right_ind, contra_left_ind, end_time, save_folder_peak, component = 'P190', text = 'P190')


#%%

import meta_functions_HC as function_hc



contra_right = {
    'P1':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'],
    'N1':['F1', 'F3', 'FC1', 'FC3'],
    'N2':ch_names_r_hc[2],
    'P2':['Cz'],
    'P3':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'] #ipsi
    }

contra_left =  {
    'P1':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'],
    'N1':['F2', 'F4', 'FC2', 'FC4'],
    'N2':ch_names_l_hc[2],
    'P2':['Cz'],
    'P3':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'] #ipsi
    }


contra_right = {
    'P1':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'],
    'N1':['F1', 'F3', 'FC1', 'FC3'],
    'N2':['F7', 'FC5', 'CP5', 'C5', 'FT7', 'TP7'],
    'P2':['Cz'],
    'P3':ch_names_r['v2'][2] #ipsi
    }

contra_left =  {
    'P1':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'],
    'N1':['F2', 'F4', 'FC2', 'FC4'],
    'N2':['F4', 'Fz', 'FC2', 'F2', 'AF4', 'FC4'],
    'P2':['Cz'],
    'P3':ch_names_l['v2'][2] #ipsi
    }



# I'm using P2 vs P3 for contra and ipsi ---> laterality index
time_windows = {'P1':[15, 50], 'N1':[46, 80], 'N2':[100, 135 ],'P2':[150, 220],'P3':[90, 135]}
amp_l_arr_st, amp_r_arr_st, lat_l_arr_st, lat_r_arr_st =  function_hc.amp_latency_6_components_st(evokeds_all_L, evokeds_all_R, contra_right, contra_left, time_windows, save_folder_peak)

lat_l_arr_st_del = lat_l_arr_st[:, 1:, :]
lat_r_arr_st_del = lat_r_arr_st[:, np.array([1, 3]), :]
amp_t_hc, lat_t_hc =  function_hc.amp_latency_6_components_hc(evokeds_all_L_hc, evokeds_all_R_hc, contra_right, contra_left, time_windows, save_folder_peak)
# amp_r ------>  (6 x 4 x 5) =  (component x sub x v2-v6) 




#%% Bar plot statistical analysis

import meta_functions_HC as function_hc




component_number = 0 # P1
p_ind_p1, fig_P30_latency = function_hc.box_plot_p1_all_time_points_HC_st(component_number, lat_t_hc, lat_l_arr_st_del, lat_r_arr_st_del, save_folder_peak)


component_number = 1 # N1
t_ind , p_ind, t_paired, p_paired, N1_latency_fig = function_hc.bar_plot_n2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak)

# Our argument is N2 is not formed
component_number = 3 # P2
t_ind , p_ind, t_paired, p_paired, P2_latency_fig = function_hc.bar_plot_p2_HC_st(component_number, lat_t_hc, lat_l_arr_st, lat_r_arr_st, save_folder_peak)




#%%

import matplotlib.image as mpimg
import io

fig, axs = plt.subplots(4, 2, figsize=(6.3, 9.5))

# Helper function to render a figure in an axes
def render_fig_to_ax(fig_obj, ax, title=None):
    buf = io.BytesIO()
    fig_obj.savefig(buf, format='png')
    buf.seek(0)
    img = mpimg.imread(buf)
    ax.imshow(img, aspect='equal')
    ax.axis('off')
    if title:
        ax.set_title(title, loc='left', y=0.90, x=0.1, fontweight = 'bold')

# Row 0
render_fig_to_ax(contra_p30, axs[0, 0], 'A')
render_fig_to_ax(fig_P30_latency, axs[0, 1], 'B')

# Row 1
render_fig_to_ax(contra_n60, axs[1, 0], 'C')
render_fig_to_ax(N1_latency_fig, axs[1, 1], 'D')

# Row 2
render_fig_to_ax(ipsi_p120, axs[2, 0], 'E')
render_fig_to_ax(fig_LI_in, axs[2, 1], 'F')
axs[2, 1].axis('off')

# Row 3
render_fig_to_ax(contra_190, axs[3, 0], 'G')
render_fig_to_ax(P2_latency_fig, axs[3, 1], 'H')
plt.tight_layout()
plt.subplots_adjust(
    left=0.03,
    right=0.97,
    top=0.98,
    bottom=0.02,
    wspace=0.01,
    hspace=0.02)
plt.show()
fig.savefig(save_folder_peak  + 'Figure_2_paper'+ '.tif', dpi = 600, pil_kwargs={'compression': 'tiff_lzw'} , overwrite = True)


#%%


contra_right = {
    'P1':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'],
    'N1':['F1', 'F3', 'FC1', 'FC3'],
    'N2':ch_names_r_hc[2],
    'P2':['Cz'],
    'P3':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'] #ipsi
    }

contra_left =  {
    'P1':['C4', 'C6', 'CP4', 'CP6', 'C2','CP2'],
    'N1':['F2', 'F4', 'FC2', 'FC4'],
    'N2':ch_names_l_hc[2],
    'P2':['Cz'],
    'P3':['C3', 'C5', 'CP3', 'CP5', 'C1', 'CP1'] #ipsi
    }


# I'm using P2 vs P3 for contra and ipsi ---> laterality index
time_windows = {'P1':[15, 50], 'N1':[46, 80], 'N2':[100, 135 ],'P2':[150, 220],'P3':[100, 135]}
amp_l_arr_st, amp_r_arr_st, lat_l_arr_st, lat_r_arr_st =  function_hc.amp_latency_6_components_st(evokeds_all_L, evokeds_all_R, contra_right, contra_left, time_windows, save_folder_peak)

lat_l_arr_st_del = lat_l_arr_st[:, 1:, :]
lat_r_arr_st_del = lat_r_arr_st[:, np.array([1, 3]), :]
amp_t_hc, lat_t_hc =  function_hc.amp_latency_6_components_hc(evokeds_all_L_hc, evokeds_all_R_hc, contra_right, contra_left, time_windows, save_folder_peak)
# amp_r ------>  (6 x 4 x 5) =  (component x sub x v2-v6) 


import meta_functions_HC as function_hc    
#% Clinical Assessment
clinical_xl =  pd.read_excel('IN-TENS_Clinical_assessments_scores_all .xlsx', sheet_name=None)

subjects_part_ids = ['AmWo', 'FuMa', 'GrMa', 'KaBe', 'SoFa', 'WiLu', 'BuUl', 'EiHe', 'GuWi', 'MeRu']

fuma_score = np.zeros([10, 5]) 

for i, i_n in enumerate(subjects_part_ids):
    
    k = list(clinical_xl[str('Sheet1')][str('part_id')]).index(f'{i_n}')
    
    fuma_score[i, :] = clinical_xl[str('Sheet1')]['fmue'].iloc[np.arange(k, 95,19)]
    
    
fuma_score_swap = fuma_score[:, [0, 2, 1, 3 , 4]]
fuma_score_swap[5, 2] = 21 # just a given value    
time_points = ['v2', 'v3', 'v4', 'v5', 'v6']












p_t = {}
r_t = {}
analysis = ['amp', 'lat']

for i_analysis, analysis in enumerate(analysis):
    p_t[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    r_t[str(analysis)] = np.zeros([len(time_windows), len(time_points)])
    for i_components, n_components in enumerate(time_windows):         
        for i_time, n_time in enumerate(time_points):   
            if analysis == 'amp':
                r_t[str(analysis)][i_components, 
                                   i_time], p_t[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_swap[:, i_time], np.concatenate((amp_l_arr_st[i_components,:, i_time ], amp_r_arr_st[i_components,:, i_time])))
            elif analysis == 'lat':   
                r_t[str(analysis)][i_components, i_time], p_t[str(analysis)][i_components, i_time] = scipy.stats.pearsonr(fuma_score_swap[:, i_time], np.concatenate((lat_l_arr_st[i_components,:, i_time ], lat_r_arr_st[i_components,:, i_time])))
                
                
                
                


function_hc.FM_UE_plotting(amp_l_arr_st, amp_r_arr_st, fuma_score_swap, save_folder_peak)
