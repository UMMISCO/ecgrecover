import numpy as np
import random
from tqdm import tqdm
from scipy import signal


def normalization(Z):
    mini=Z.min()
    maxi=Z.max()
    return(-1+((Z-mini)*(2))/(maxi-mini))
    
def filtering(Z):
    mini=Z.min()
    maxi=Z.max()
    signal_bef = np.zeros((len(Z[:])))
    temp = (-1+((Z-mini)*(2))/(maxi-mini))
    signal_bef = temp
    nyquist = 0.5 * 500  
    low_cutoff = 0.05 / nyquist
    high_cutoff = 150.0 / nyquist
    new_sampling_frequency = 512
    original_sampling_frequency = 5000
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    filtered_signal = signal.lfilter(b, a, signal_bef)
    filtered_signal = signal_bef
    resampled_signal = normalization(signal.resample(filtered_signal, int(len(filtered_signal) * (new_sampling_frequency / original_sampling_frequency))))
    if np.all(np.isnan(resampled_signal)):
        resampled_signal = np.random.normal(0,1,(512))
    return(resampled_signal)


def define_possibility(data):
    P = []
    L = []
    for i in range(1,12):
        if 12%i == 0:
            sous_P = []
            sous_L = []
            for j in range(0,12,i):
                sous_L.append(j)
            L.append(sous_L)
            for j in range(0,len(data[0])+1, int(len(data[0]) / (12 / i))):
                sous_P.append(j)
            sous_P[-1] = 512
            P.append(sous_P)
    return(L,P)

def masking(signal, L, P):
    full_cut_signal = []
    full_full_signal =[]
    for p in range(len(P)):
        cut_signal = np.random.random(signal.shape)
        itp = 0
        it = 0
        for l in range(12):
            try:
                if L[p][it] <= l < L[p][it+1]:
                    pass
                else:
                    itp += 1
                    it  +=1
            except IndexError:
                pass
            cut_signal[P[p][itp]:P[p][itp+1],l] = signal[P[p][itp]:P[p][itp+1],l]
            signal[:,l] = signal[:,l]
        full_cut_signal.append(np.array(cut_signal))
        full_full_signal.append(signal)

    for i in range(12):
        cut_signal = np.random.random(signal.shape)
        cut_signal[:,i] = signal[:,i]
        full_cut_signal.append(np.array(cut_signal))
        full_full_signal.append(signal)
    full_cut_signal = np.array(full_cut_signal)
    full_full_signal = np.array(full_full_signal)
    return(full_cut_signal, full_full_signal)
    
def creat_dataset(data):
    cut_data = []
    full_data = []
    L,P = define_possibility(data)
    for i in tqdm(range(len(data))):
        cut, full = masking(data[i], L,P)
        cut_data.append(cut)
        full_data.append(full)
    cut_data = np.array(cut_data)
    full_data = np.array(full_data)
    return(np.concatenate(cut_data,0), np.concatenate(full_data,0))


def pre_processing(path, seed):
    data = np.load(path)
    new_data = np.zeros((len(data),512,12))
    for i in tqdm(range(len(data))):
        for l in range(12):
            new_data[i,:,l] = filtering(data[i,:,l])
    data_mask, data_true = creat_dataset(new_data)
    np.random.seed(seed)
    np.random.shuffle(data_mask)
    np.random.seed(seed)
    np.random.shuffle(data_true)
    return(data_mask, data_true)