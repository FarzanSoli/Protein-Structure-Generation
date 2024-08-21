import torch
import pickle
import numpy as np
from itertools import product
from torch.utils.data import DataLoader
from Training_Model import Training_Model
from Density_Coverage import compute_prdc
from Functions import Frechet_distance, Sampling
from Config import config, positional_embedding, load_checkpoint
from Functions import Numpy_normalize, normalize_coordinates, CustomDataset
from Functions import align_data_with_ground_truth, compute_reordered_coordinate
# ========================================== #
class Protein_Generation:
    def __init__(self, Samples = 1000, eta = 1):
        super(Protein_Generation, self).__init__()
        self.eta = eta
        self.Samples = Samples
        self.device = config().device
        # Define the grid of hyperparameters
        self.param_grid = {
            'num_epochs': [1, 3, 5],
            'Data_Aug_Folds': [1, 10, 20],
            'lr': [1e-6, 5e-7, 1e-7],
            'batch_size': [64, 128, 256],
            }
        # ------------------------------------------ #
        # Perform grid search
        best_loss = float('inf')
        best_params = None
        best_model = None
        # ------------------------------------------ #
        for params in product(*self.param_grid.values()):
            param_dict = dict(zip(self.param_grid.keys(), params))
            # -------------------------------------- #
            self.trainer = Training_Model(
                num_epochs=param_dict['num_epochs'],
                Data_Aug_Folds=param_dict['Data_Aug_Folds'],
                lr=param_dict['lr'],
                batch_size=param_dict['batch_size']
            )
            final_loss, self.trained_model, optimizer = self.trainer.train()

            if final_loss < best_loss:
                best_loss = final_loss
                best_params = param_dict
                best_model = self.trained_model
        print(f'Best parameters found: {best_params} with loss: {best_loss}')
        # ========================================== #
        ### Load the latest checkpoint
        # latest_checkpoint = 'checkpoints/checkpoint_epoch_X.pth'  # Replace X with the desired epoch
        # model, optimizer, start_epoch, loss = load_checkpoint(self.trained_model, optimizer, latest_checkpoint)
        # ============================================
        # eta = 0 --> DDIM 
        # eta = 1 --> DDPM 
        X_samples, H_samples = Sampling(self.trained_model, self.Samples, self.eta, device = self.device)
        # ------------------------------------------ #
        #     Order based on positional embedding    #
        # ------------------------------------------ #
        pos_embed = positional_embedding()
        order = np.argsort(np.sum(pos_embed.detach().cpu().numpy(), axis=1))
        # ------------------------------------------ #
        X_samples_ordered = []
        for i in range(X_samples.shape[0]):
            X_samples_ordered.append(normalize_coordinates(
                compute_reordered_coordinate(X_samples.cpu().numpy()[i,:,:], order)))
        X_samples_ordered = np.array(X_samples_ordered)
        torch.save(X_samples_ordered, 'X_samples_ordered.pt')
        # ------------------------------------------ #
        H_samples_ordered = []
        for i in range(H_samples.shape[0]):
            H_samples_ordered.append(Numpy_normalize(
                compute_reordered_coordinate(H_samples.cpu().numpy()[i,:,:], order)))
        H_samples_ordered = np.array(H_samples_ordered)
        torch.save(H_samples_ordered, 'H_samples_ordered.pt')
        # ========================================== #
        #            Perfomance evaluation           #
        # ========================================== #
        real_coordinates = []
        real_coordinates_reshaped = []
        aligned_real, aligned_gen = [], []
        Frechet_dist, density_coverage = [], []
        # ------------------------------------------ #
        real_features = []
        Frechet_dist_features = []
        real_features_reshaped = []
        density_coverage_features = []
        # ------------------------------------------ #
        with open('Dataset/Test_32.pkl', 'rb') as file:
            Test_dataset = pickle.load(file)
        # =============================================
        num_real_instances = X_samples.shape[0]
        for i in range(len(DataLoader(CustomDataset(Test_dataset), 
                               num_real_instances, shuffle=True))):
            # -----------------------------------------
            real_coordinates.append(next(iter(DataLoader(CustomDataset(Test_dataset), 
                                   num_real_instances, shuffle=True)))[0][:,:config().num_residues,:])
            real_coordinates_reshaped.append(
                                   real_coordinates[-1].reshape(-1, real_coordinates[-1].shape[-1]))
            # --------------------------------------- #
            #     Aligning coordinates using PCA      #
            # --------------------------------------- #
            aligned_real_, aligned_gen_ = align_data_with_ground_truth(
                                                                    real_coordinates[-1].cpu().numpy(), 
                                                                    X_samples_ordered)
            aligned_real.append(aligned_real_)
            aligned_gen.append(aligned_gen_)
            # -----------------------------------------
            Frechet_dist.append(Frechet_distance(aligned_real[-1], aligned_gen[-1]))
            # -----------------------------------------
            density_coverage.append(compute_prdc(
                                    real_instances = aligned_real[-1],
                                    generated_instances = aligned_gen[-1], 
                                    nearest_k=5))
            # =========================================
            real_features.append(next(iter(DataLoader(CustomDataset(Test_dataset), 
                                   num_real_instances, shuffle=True)))[1][:,:config().num_residues,:])
            real_features_reshaped.append(real_features[-1].reshape(-1, real_features[-1].shape[-1]))
            # -----------------------------------------
            H_samples_ordered_reshaped = H_samples_ordered.reshape(-1, H_samples_ordered.shape[-1])
            Frechet_dist_features.append(
                                    Frechet_distance(real_features_reshaped[-1], 
                                    H_samples_ordered_reshaped))
            # -----------------------------------------
            real_features_reshaped.append(
                                        real_features[-1].reshape(real_features[-1].shape[0], 
                                        real_features[-1].shape[1]*real_features[-1].shape[-1]))
            H_samples_ordered_reshaped = H_samples_ordered.reshape(H_samples_ordered.shape[0], 
                                                           H_samples_ordered.shape[1]*H_samples_ordered.shape[-1])
            density_coverage_features.append(compute_prdc(
                                            real_instances = real_features_reshaped[-1],
                                            generated_instances = H_samples_ordered_reshaped, 
                                            nearest_k=5))
        # =============================================
        density_coordinates = np.max([x['density'] for x in density_coverage])
        print(f"Density of Coordinates: {density_coordinates}")
        coverage_coordinates = np.max([x['coverage'] for x in density_coverage])
        print(f"Coverage of Coordinates: {coverage_coordinates}")
        density_features = np.max([x['density'] for x in density_coverage_features])
        print(f"Density of Features: {density_features}")
        coverage_features = np.max([x['coverage'] for x in density_coverage_features])
        print(f"Coverage of Features: {coverage_features}")
        Frechet_Coordinates = np.min(Frechet_dist)
        print(f"Frechet Distance of Coordinates: {Frechet_Coordinates}")
        Frechet_features = np.min(Frechet_dist_features)
        print(f"Frechet Distance of Features: {Frechet_features}")
        # =============================================



Protein_Generation(Samples = 1000, eta = 1)
