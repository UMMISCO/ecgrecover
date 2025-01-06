from tools.PreProcesing import pre_processing
from tools.LoadModel import load_model
from tools.CompleteData import Complete_data
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from tools.SoftDTW_functions import SoftDTW



def MAE(x,y):
    MAE = np.mean(np.absolute(x - y))
    return(MAE)
def RMSE(x,y):
    RMSE = np.sqrt(np.mean((x-y)**2))
    return(RMSE)

def cut_lead(r,t,lead, config):
    list_config = np.array([[[0,42],[42,84],[84,126],[126,168],[168,210],[210,252],[252,294],[294,336],[336,378],[378,420],[420,462],[462,512]],
                  [[0,85],[0,85],[85,170],[85,170],[170,255],[170,255],[255,340],[255,340],[340,425],[340,425],[425,512],[425,512]],
                  [[0,128],[0,128],[0,128],[128,256],[128,256],[128,256],[256,384],[256,384],[256,384],[384,512],[384,512],[384,512]],
                  [[0,170],[0,170],[0,170],[0,170],[170,340],[170,340],[170,340],[170,340],[340,512],[340,512],[340,512],[340,512]],
                  [[0,256],[0,256],[0,256],[0,256],[0,256],[0,256],[256,512],[256,512],[256,512],[256,512],[256,512],[256,512]]])
    end1, start2 = list_config[config][lead]
    new_r1 = r[:end1]
    new_r2 = r[start2:]
    new_r = np.concatenate([new_r1, new_r2])
    new_t1 = t[:end1]
    new_t2 = t[start2:]
    new_t = np.concatenate([new_t1, new_t2])
    return(new_r, new_t)


def calculate_all(config_regenerate, True_):
    dic_lead = {0:"I",1:"II",2:"III",3:"AVR",4:"AVL",5:"AVF",6:"V1",7:"V2",8:"V3",9:"V4",10:"V5",11:"V6"}
    sdtw = SoftDTW(use_cuda=False, gamma=0.1)
    liste_corr = []
    liste_RMSE = []
    liste_MAE = []
    liste_ecg = []
    liste_lead = []
    liste_config = []
    liste_dtw = []
    liste_sot = []
    for ecg in tqdm(range(len(config_regenerate[0]))):
        for lead in range(12):
            for config in range(len(config_regenerate)):
                
                liste_ecg.append(ecg)
                liste_lead.append(dic_lead[lead])
                liste_config.append(config)

                ### Cut Lead to calculate only on the segment reconstruction ###
                if config < 5 :
                    r, t = cut_lead(config_regenerate[config,ecg,lead,:], True_[config,ecg,lead,:],lead, config )
                    rr = torch.tensor(np.expand_dims(np.expand_dims(r,0),1))
                    tt = torch.tensor(np.expand_dims(np.expand_dims(t,0),1))
                    
                    liste_dtw.append(np.round(sdtw(rr,tt)[0].numpy()/len(r),3))
                    liste_corr.append(np.round(pearsonr(r,t)[0],3))
                    liste_MAE.append(np.round(MAE(r,t),3))
                    liste_RMSE.append(np.round(RMSE(r,t),3))


                else:
                    r, t = config_regenerate[config,ecg,lead,:], True_[config,ecg,lead,:]
                    rr = torch.tensor(np.expand_dims(np.expand_dims(r,0),1))
                    tt = torch.tensor(np.expand_dims(np.expand_dims(t,0),1))
                    liste_dtw.append(np.round(sdtw(rr,tt)[0].numpy()/len(r),3))
                    
                    liste_corr.append(np.round(pearsonr(config_regenerate[config,ecg,lead,:],True_[config,ecg,lead,:])[0],3))
                    liste_MAE.append(np.round(MAE(config_regenerate[config,ecg,lead,:],True_[config,ecg,lead,:]),3))
                    liste_RMSE.append(np.round(RMSE(config_regenerate[config,ecg,lead,:],True_[config,ecg,lead,:]),3))

    colonnes = ['ECG','Lead','Config','PCC', 'RMSE', 'MAE', 'DTW']
    df = pd.DataFrame(np.array([liste_ecg,liste_lead,liste_config,
                            liste_corr, liste_RMSE, liste_MAE, liste_dtw]).T, columns=colonnes)
    return(df)


def Compute_metrics(path_data, path_model, path_save, sd, dvc):
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
    
    df = calculate_all(config_regenerate, config_True)
    df.to_csv(path_save+"Metrics_ECGrecover.csv")
    


    
    

    
