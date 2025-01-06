import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from tools.LoadModel import load_model
from tools.PreProcesing import pre_processing
from tools.LossFunction import loss_function



### Load Data ###

def load_Data(path, s, Verbose = False):
    X_train, X_true = pre_processing(path+"train.npy",seed = s)
    X_validation, X_validation_true = pre_processing(path+"validation.npy",seed = s)

    #print("Train data: ", X_train.shape)
    #print("True data: ", X_true.shape)
    #print("\n")
    #print("Validation data: ", X_validation.shape)
    #print("Validation True data: ", X_validation_true.shape)
    return(X_train, X_true,X_validation, X_validation_true)


### Train ###

def training(train_set, train_label, validation_set, validation_label, epoch, batch_size, Autoencoder, device, optimizer, scheduler, loss, save_path=''):
    lost_list_train = []
    lost_list_validation =  []
    corr_list_validation = []
    corr_list_train = []
    mse_list_validation = []
    mse_list_train = []
    for e in range(epoch):
        start_time = time.time()
        train_loss = 0.
        validation_loss = 0.
        corr_train = 0.
        corr_validation = 0.
        mse_train = 0.
        mse_validation = 0.
        prec = 0
        it = 0
        for b in range(0,len(train_set), batch_size):
            data = train_set[b:b + batch_size].astype("float32") 
            data = np.expand_dims(data,1)
            label = np.array(train_label[b:b + batch_size,:,:])
            data = torch.tensor(data).to(device)
            data = torch.transpose(data,2,3)
            label = torch.tensor(label).to(device)
            label = torch.transpose(label,1,2)
            
            Autoencoder.zero_grad()
            Xc = Autoencoder(data, device)
            lossMSE, corr, mse  = loss(label,Xc)
            lossMSE.backward()
            optimizer.step()
            
            # update poarameters for Discriminator net
            corr_train += corr.to('cpu').detach().numpy()
            #corr_train += corr
            mse_train += mse.to('cpu').detach().numpy()
            train_loss += lossMSE.to('cpu').detach().numpy()
            it+=1
            
        lost_list_train.append(train_loss/it)
        corr_list_train.append(corr_train/it)
        mse_list_train.append(mse_train/it)
        scheduler.step(train_loss)
        it2 = 0 
        for b in range(0,len(validation_set), batch_size):
            data = validation_set[b:b + batch_size].astype("float32") 
            data = np.expand_dims(data,1)
            label = np.array(validation_label[b:b + batch_size,:,:])
            data = torch.tensor(data).to(device)
            data = torch.transpose(data,2,3)
            label = torch.tensor(label).to(device)
            label = torch.transpose(label,1,2)
            
            Xc = Autoencoder(data, device)
            
            
            lossMSE_validation, corr, mse = loss(label,Xc)
            validation_loss += lossMSE_validation.to('cpu').detach().numpy()
            corr_validation += corr.to('cpu').detach().numpy()
            #corr_test += corr
            mse_validation += mse.to('cpu').detach().numpy()
            it2+=1
        scheduler.step(lossMSE_validation)

            
        lost_list_validation.append(validation_loss)
        corr_list_validation.append(corr_validation)
        mse_list_validation.append(mse_validation)
        print("Epoch %d: lossTrain: %.3f  lossValidation: %.3f corr_Train: %.3f corr_Validation: %.3f mse_Train: %.3f mse_Validation: %.3f Time: %.3f sec" % 
              (e, train_loss/it,
               validation_loss/it2,
               corr_train/it, 
               corr_validation/it2,
               mse_train/it, 
               mse_validation/it2,
               time.time() - start_time))
        
    if save_path != '':
        torch.save(Autoencoder.state_dict(), save_path+"Modeltemp.pth")
        np.save(save_path+"lost_list_train.npy",lost_list_train)
        np.save(save_path+"lost_list_validation.npy",lost_list_validation)
        np.save(save_path+"corr_list_train.npy",corr_list_train)
        np.save(save_path+"corr_list_validation.npy",corr_list_validation)
        np.save(save_path+"mse_list_train.npy",mse_list_train)
        np.save(save_path+"mse_list_validation.npy",mse_list_validation)
    
    #return(lost_list_train, lost_list_validation, corr_list_train, corr_list_validation,mse_list_train,mse_list_validation)


def Train_model(path, sd, dvc, epoch, batch_size, save_path = ''):

    ### Load Data ###
    print("Load Data")
    X_train, X_true,X_validation, X_validation_true = load_Data(path,sd)

    ### Load Model on GPU ###
    print("Load Model")
    Autoencoder = load_model()
    device = torch.device(dvc)
    Autoencoder = Autoencoder.to(device)

    ### Load loss function ###
    loss_f = loss_function
    loss = loss_f()

    ### Optimizer ###
    optimizer = optim.Adam(Autoencoder.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    print("Train Model")
    training(X_train, X_true, X_validation, X_validation_true, epoch, batch_size, Autoencoder, device, optimizer, scheduler, loss, save_path)
