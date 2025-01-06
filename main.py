from learn.Training import Train_model
from compute_metrics.compute_metrics import Compute_metrics
from compute_metrics.compute_peaks import Compute_peaks
import argparse



def main(data_path, save_path, seed, device, Train, save_results = "Results/", epoch = 10, batch_size = 256):
    if Train == True:
        print("Train Model")
        Train_model(data_path, seed, device, epoch, batch_size, save_path)
    else:
        print("Compute Metrics")
        Compute_metrics(data_path,save_path+"/Model.pth", save_results, seed, device)
        print("\nCompute Peaks")
        Compute_peaks(data_path,save_path+"/Model.pth", save_results, seed, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='program for training ECGrecover on PTB-XL base')
    parser.add_argument('data_path', type=str, help='Path to data (precise file for calculating metrics / training folder)')
    parser.add_argument('save_path', type=str, help='Path where model is to be saved (Training) or where model is located (calculates metrics)')
    parser.add_argument('seed', type=int, help='Seed for data mixing')
    parser.add_argument('device', type=int, help='GPU to be used')
    
    parser.add_argument('--Train', action='store_true', help='parameter to specify if you want to train the model')

    parser.add_argument('save_results', type=str, help='Where metrics tables should be saved')
    
    parser.add_argument('epoch', nargs='?', type=int, default = 10, help='Number of epochs for model training')
    parser.add_argument('batch_size', nargs='?', type=int, default = 256, help='Batch size for model training')
    parser.add_argument('--Verbose', action='store_true', help='Verbose for the training')

    args = parser.parse_args()
    main(args.data_path, args.save_path, args.seed, args.device, args.Train, args.save_results , args.epoch, args.batch_size)
