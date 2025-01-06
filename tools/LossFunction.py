import torch
import torch.nn as nn

class RMSE_Loss(torch.nn.Module):
    def __init__(self, alpha=1):
        super(RMSE_Loss, self).__init__()

    def forward(self, y_true, y_pred):
        loss1 = torch.mean(torch.square(y_true - y_pred), axis = 2)
        loss1= torch.nan_to_num(loss1)
        loss2 = pearson_correlation(y_true, y_pred)
        loss = loss1 - (0.1*loss2)
        return(torch.mean(torch.mean(loss,axis =1)), torch.mean(loss2), torch.mean(loss1))
RMSELoss = RMSE_Loss()

def pearson_correlation(x,y):
    # Ensure that x and y are the same length
    assert len(x) == len(y)

    # Calculate the means of x and y
    x_mean = torch.mean(x, axis = 2)
    y_mean = torch.mean(y, axis = 2)

    # Calculate the variance of x and y
    x_variance = torch.var(x, axis = 2)
    y_variance = torch.var(y, axis = 2)

    # Calculate the standard deviations of x and y
    x_stddev = torch.sqrt(x_variance)
    y_stddev = torch.sqrt(y_variance)



    # Calculate the Pearson correlation coefficient
    r = torch.sum((x - torch.unsqueeze(x_mean, 2)) * (y - torch.unsqueeze(y_mean, 2)), axis = 2 )
    r = r/((len(x[0][0]) * x_stddev * y_stddev))
    r = torch.nan_to_num(r)

    return r


class loss_function(torch.nn.Module):
    def __init__(self, alpha=1):
        super(loss_function, self).__init__()

    def forward(self, y_true, y_pred):
        # Calculate the MSE
        loss1 = torch.mean(torch.square(y_true - y_pred), axis = 2)

        # Calculate the mean correlation of an ECG
        loss2 = pearson_correlation(y_true, y_pred)
        
        # Calculate the loss 
        loss = loss1 - (0.1*loss2)

        # Return the loss, the pearson correlation and the MSE
        return(torch.mean(torch.mean(loss,axis =1)), torch.mean(loss2), torch.mean(loss1))
