import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class Convolution1D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Convolution1D_layer, self).__init__()
        self.f = out_f
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_f, out_channels=out_f, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2)
        )

        
    def forward(self, x, device):
        b = len(x)
        new_x = torch.tensor(np.zeros((b,self.f, 12, int(x.shape[-1]/2))).astype("float32")).to(device)
        for i in range(12):
            new_x[:,:,i,:] = self.conv(x[:,:,i,:])
        return(new_x)


class Deconvolution1D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Deconvolution1D_layer, self).__init__()
        self.f = out_f
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_f, out_channels=out_f, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2)
        )

        
    def forward(self, x, device):
        b = len(x)
        new_x = torch.tensor(np.zeros((b,self.f, 12, int(x.shape[-1]*2))).astype("float32")).to(device)
        for i in range(12):
            new_x[:,:,i,:] = self.deconv(x[:,:,i,:])
        return(new_x)
        
class Convolution2D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Convolution2D_layer, self).__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size = (13,4), stride = (1,2), padding = (6,1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
            #nn.Dropout(0.2)
        )

        
    def forward(self, x):
        new_x = self.conv(x)
        return(new_x)       


class Deconvolution2D_layer(nn.Module):
    def __init__(self, in_f, out_f):
        super(Deconvolution2D_layer, self).__init__()
        self.f = out_f
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_f, out_channels=out_f, kernel_size = (13,4), stride = (1,2), padding = (6,1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
            #nn.Dropout(0.2)
        )

        
    def forward(self, x):
        new_x = self.deconv(x)
        return(new_x)  



class Autoencoder_net(nn.Module):
    def __init__(self):
        super(Autoencoder_net, self).__init__()
        self.first_conv2D = Convolution2D_layer(1,16)
        self.first_conv1D = Convolution1D_layer(1,16)

        self.second_conv2D = Convolution2D_layer(16,32)
        self.second_conv1D = Convolution1D_layer(16,32)

        self.third_conv2D = Convolution2D_layer(32,64)
        self.third_conv1D = Convolution1D_layer(32,64)

        self.fourth_conv2D = Convolution2D_layer(64,128)
        self.fourth_conv1D = Convolution1D_layer(64,128)

        self.first_deconv1D = Deconvolution1D_layer(256,128)
        self.first_deconv2D = Deconvolution2D_layer(256,128)

        self.second_deconv1D = Deconvolution1D_layer(256,64)
        self.second_deconv2D = Deconvolution2D_layer(256,64)

        self.third_deconv1D = Deconvolution1D_layer(128,32)
        self.third_deconv2D = Deconvolution2D_layer(128,32)

        self.fourth_deconv1D = Deconvolution1D_layer(64,1)
        self.fourth_deconv2D = Deconvolution2D_layer(64,1)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size = (13,3), stride = (1,1), padding = (6,1)),
            nn.Tanh(),
        )

        self.transition_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size = (13,3), stride = (1,1), padding = (6,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02)
        )
        

        
    def forward(self, x, device):
        conv2D_1 = self.first_conv2D(x)
        conv1D_1 = self.first_conv1D(x, device)
        conv_1 = torch.concat((conv1D_1,conv2D_1),axis = 1)
        #print("Conv1: ",conv1D_1.shape)
        #print("Conv1: ",conv2D_1.shape)

        conv2D_2 = self.second_conv2D(conv2D_1)
        conv1D_2 = self.second_conv1D(conv1D_1,device)
        #print("Conv2: ",conv1D_2.shape)
        #print("Conv2: ",conv2D_2.shape)
        conv_2 = torch.concat((conv1D_2,conv2D_2),axis = 1)
        #print("Conv2: ",conv_2.shape)

        conv2D_3 = self.third_conv2D(conv2D_2)
        conv1D_3 = self.third_conv1D(conv1D_2,device)
        #print("Conv3: ",conv1D_3.shape)
        #print("Conv3: ",conv2D_3.shape)
        conv_3 = torch.concat((conv1D_3,conv2D_3),axis = 1)
        #print("Conv3: ",conv_3.shape)

        conv2D_4 = self.fourth_conv2D(conv2D_3)
        conv1D_4 = self.fourth_conv1D(conv1D_3,device)
        #print("Conv4: ",conv1D_4.shape)
        #print("Conv4: ",conv2D_4.shape)
        conv_4 = torch.concat((conv1D_4,conv2D_4),axis = 1)
        #print("Conv4: ",conv_4.shape)

        transition = self.transition_block(conv_4)
        #print("Transition: ", transition.shape)


        deconv2D_1 = self.first_deconv2D(conv_4)
        #print("Deconv 1: ",deconv2D_1.shape)
        deconv_1 = torch.concat((deconv2D_1,conv_3),axis = 1)
        #print("Deconv 1 Concat: ",deconv_1.shape)


        deconv2D_2 = self.second_deconv2D(deconv_1)
        #print("Deconv 2: ",deconv2D_2.shape)
        deconv_2 = torch.concat((deconv2D_2,conv_2),axis = 1)
        #print("Deconv 2 Concat: ",deconv_2.shape)

        deconv2D_3 = self.third_deconv2D(deconv_2)
        #print("Deconv 3: ",deconv2D_3.shape)
        deconv_3 = torch.concat((deconv2D_3,conv_1),axis = 1)
        #print("Deconv 3 Concat: ",deconv_3.shape)

        deconv2D_4 = self.fourth_deconv2D(deconv_3)
        #print("Deconv 4: ",deconv2D_4.shape)

        out = self.final_conv(deconv2D_4)
        out = torch.squeeze(out,1)
        return(out)



def load_model():
    Autoencoder = Autoencoder_net()
    return(Autoencoder)