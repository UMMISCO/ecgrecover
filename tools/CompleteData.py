import torch
import numpy as np

def Complete_data(data,Autoencoder,device):
    batch_size = 256
    regenerate_lead = []

    for b in range(0,len(data),batch_size):
        r = data[b:b+batch_size,:,:]
        r = np.expand_dims(r,1).astype("float32")
        r = torch.tensor(r).to(device)
        r = Autoencoder(r, device).to('cpu').detach().numpy()[:,:,:]
        regenerate_lead.append(r)
    regenerate_lead = np.array(np.concatenate(regenerate_lead))
    return(regenerate_lead)