from tools.PreProcesing import pre_processing
from tools.LoadModel import load_model
from tools.CompleteData import Complete_data
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import neurokit2 as nk

def Calculate_peaks(config_regenerate, True_):
    list_config_r = []
    list_lead_r = []
    list_ecg_r = []
    list_heartbeat_r = []
    list_p_r = []
    list_q_r = []
    list_r_r = []
    list_s_r = []
    list_toff_r = []
    
    list_pamp_r = []
    list_qamp_r = []
    list_ramp_r = []
    list_samp_r = []
    list_toffamp_r = []
    
    
    
    
    for ecg in tqdm(range(len(config_regenerate[0]))):
        for l in range (12):
            for c in range(17):
                temp_true = True_[c,ecg,l,:]
                temp_regenerated = config_regenerate[c,ecg,l,:]
                _, rpeaks_True = nk.ecg_peaks(temp_true, sampling_rate=50)
                if len(rpeaks_True['ECG_R_Peaks']) != 0:
                    for b in rpeaks_True['ECG_R_Peaks']:
                        try:
                            regenerated_heartbeat = temp_regenerated[b-20:b+30]
                            if len(regenerated_heartbeat ) == 50:
                                regenerated_list = np.concatenate([regenerated_heartbeat,regenerated_heartbeat,regenerated_heartbeat,
                                                                   regenerated_heartbeat])
                                _, rpeaks_regenerated = nk.ecg_peaks(regenerated_list, sampling_rate=50)
                                _, waves_peak_regenerated = nk.ecg_delineate(regenerated_list, rpeaks_regenerated, 
                                                                             sampling_rate=50, method="dwt")
        
                                try:
                                    p_regenerated = waves_peak_regenerated["ECG_P_Peaks"][1] 
                                except Exception as e:
                                    p_regenerated = np.nan
        
                                try:
                                    q_regenerated = waves_peak_regenerated["ECG_Q_Peaks"][1] 
                                except Exception as e:
                                    q_regenerated = np.nan
        
                                try:
                                    r_regenerated = rpeaks_regenerated["ECG_R_Peaks"][1] 
                                except Exception as e:
                                    r_regenerated = np.nan
        
                                try:
                                    s_regenerated = waves_peak_regenerated["ECG_S_Peaks"][1] 
                                except Exception as e:
                                    s_regenerated = np.nan
        
                                try:
                                    toff_regenerated = waves_peak_regenerated["ECG_T_Peaks"][1] 
                                except Exception as e:
                                    toff_regenerated = np.nan
        
                                #### Amplitude ###
                                try:
                                    pamp_regenerated = regenerated_list[waves_peak_regenerated["ECG_P_Peaks"][1]] 
                                except Exception as e:
                                    pamp_regenerated = np.nan
        
                                try:
                                    qamp_regenerated = regenerated_list[waves_peak_regenerated["ECG_Q_Peaks"][1]] 
                                except Exception as e:
                                    qamp_regenerated = np.nan
        
                                try:
                                    ramp_regenerated = regenerated_list[rpeaks_regenerated["ECG_R_Peaks"][1]] 
                                except Exception as e:
                                    ramp_regenerated = np.nan
        
                                try:
                                    samp_regenerated = regenerated_list[waves_peak_regenerated["ECG_S_Peaks"][1]] 
                                except Exception as e:
                                    samp_regenerated = np.nan
        
                                try:
                                    toffamp_regenerated = regenerated_list[waves_peak_regenerated["ECG_T_Offsets"][1]]
                                except Exception as e:
                                    toffamp_regenerated = np.nan
                                    
        
        
            
                                list_config_r.append(c)
                                list_lead_r.append(l)
                                list_ecg_r.append(ecg)
                                list_heartbeat_r.append(b)
                                list_p_r.append(p_regenerated)
                                list_q_r.append(q_regenerated)
                                list_r_r.append(r_regenerated)
                                list_s_r.append(s_regenerated)
                                list_toff_r.append(toff_regenerated)
        
                                list_pamp_r.append(pamp_regenerated)
                                list_qamp_r.append(qamp_regenerated)
                                list_ramp_r.append(ramp_regenerated)
                                list_samp_r.append(samp_regenerated)
                                list_toffamp_r.append(toffamp_regenerated)
                                
                        except Exception as e:
                            list_config_r.append(c)
                            list_lead_r.append(l)
                            list_ecg_r.append(ecg)
                            list_heartbeat_r.append(b)
                            list_p_r.append(np.nan)
                            list_q_r.append(np.nan)
                            list_r_r.append(np.nan)
                            list_s_r.append(np.nan)
                            list_toff_r.append(np.nan)
                            list_pamp_r.append(np.nan)
                            list_qamp_r.append(np.nan)
                            list_ramp_r.append(np.nan)
                            list_samp_r.append(np.nan)
                            list_toffamp_r.append(np.nan)
                else:
                    list_config_r.append(c)
                    list_lead_r.append(l)
                    list_ecg_r.append(ecg)
                    list_heartbeat_r.append(b)
                    list_p_r.append(np.nan)
                    list_q_r.append(np.nan)
                    list_r_r.append(np.nan)
                    list_s_r.append(np.nan)
                    list_toff_r.append(np.nan)
                    list_pamp_r.append(np.nan)
                    list_qamp_r.append(np.nan)
                    list_ramp_r.append(np.nan)
                    list_samp_r.append(np.nan)
                    list_toffamp_r.append(np.nan)
        
    
                       
                    
    
    
                
    colonnes = ['Config','Lead','ECG', 'HeartBeat', 'Ppos', 'Qpos', 'Rpos', 'Spos', 'Toffpos',
               'Pamp', 'Qamp', 'Ramp', 'Samp', 'Toffamp']
    df = pd.DataFrame(np.array([list_config_r,list_lead_r,list_ecg_r, list_heartbeat_r,
                                list_p_r, list_q_r, list_r_r, list_s_r, list_toff_r,
                               list_pamp_r, list_qamp_r, list_ramp_r, list_samp_r, list_toffamp_r]).T, columns=colonnes)
    return(df)



def Compute_peaks(path_data, path_model, path_save, sd, dvc):
    ### Load Testing data ###
    X_test, X_test_true = pre_processing(path_data,seed = sd)
    X_test_true = np.swapaxes(X_test_true,1,2)
    X_test = np.swapaxes(X_test,1,2)
    
    ### Load Trained Model ###
    Autoencoder = load_model()
    weight = torch.load(path_model,map_location=torch.device('cpu'))
    Autoencoder.load_state_dict(weight)
    device = torch.device(dvc)
    Autoencoder = Autoencoder.to(device)

    ### Regenerate lead ###
    X_regenerated = Complete_data(X_test, Autoencoder, device)
    
    ### Split the dataset in 17 configuration
    config_regenerate = np.zeros((17,int(len(X_test)/17),12,512))
    config_True = np.zeros((17,int(len(X_test_true)/17),12,512))
    
    it = 0
    ecg = 0
    for i in range(len(X_test)):
        config_regenerate[it,ecg,:,:] = X_regenerated[i,:,:]
        it += 1
        if it == 17:
            it = 0
            ecg +=1
    it = 0
    ecg = 0
    for i in range(len(X_test_true)):
        config_True[it,ecg,:,:] = X_test_true[i,:,:]
        it += 1
        if it == 17:
            it = 0
            ecg +=1

    df = Calculate_peaks(config_regenerate, config_True)
    df.to_csv(path_save+ "Peaks_ECGrecover.csv")


