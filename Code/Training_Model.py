import os
import torch
import pickle
import numpy as np
from EGNN import *
from tqdm import tqdm
from Config import config
from torch.utils.data import DataLoader
from Training_Denoizer import Noise_Pred
from Data_Processing import Data_Processing
from Functions import CustomDataset, dynamic_weighting
# ============================================
class Training_Model():
    # ========================================== #
    def __init__(self, device = torch.device('cuda:0'), num_epochs = 1,
                 pad_length = 32, Data_Aug_Folds = 1):
        super(Training_Model, self).__init__()
        self.device = device
        self.num_epochs = num_epochs
        self.pad_length = pad_length
        self.Data_Aug_Folds = Data_Aug_Folds
        self.device = torch.device('cuda:0')
        self.length = config().num_residues
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        self.log_var_x = nn.Parameter(torch.zeros(()))
        self.log_var_f = nn.Parameter(torch.zeros(()))
        # ------------- Load Dataset ------------- #
        if not os.path.exists('Dataset/Train_32.pkl'):
            Data_Processing(self.pad_length, self.Data_Aug_Folds).Data_Augmentation()
    # ========================================== #
    def train(self):
        # -------------------------------------- #
        with open('Dataset/Train_32.pkl', 'rb') as file:
            train_data = pickle.load(file)
        self.Train_data_loader = DataLoader(CustomDataset(train_data), 
                                       config().batch_size, shuffle=True)
        # ------------ Import Denoizer ----------- #
        self.model = Noise_Pred()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 5e-7)
        # -------------------------------------- #
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for data in tqdm(self.Train_data_loader):
                coordinates = data[0][:, :self.length, :].float().to(device=self.device)
                features = data[1][:, :self.length, :].float().to(device=self.device)     
                # Compute loss
                loss_x, loss_f = self.model.loss_fn(coordinates, features)
                loss_x = loss_x.detach()
                loss_f = loss_f.detach()
                weighted_loss = dynamic_weighting(loss_x, loss_f, self.log_var_x, self.log_var_f)       
                # Zero gradients, perform a backward pass, and update the weights
                optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
                optimizer.step()
                total_loss += weighted_loss.item() 
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(self.Train_data_loader)}')
        return self.model
        
Training_Model().train()


# =============================================================================
# 
# 
# 
# """ ################## Sampling (Generation) ################## """    
# X_samples, H_samples = Sampling(model, device=device, Samples = 1000, eta = 1)
# # ------------------------------------------ #
# #     Order based on positional embedding    #
# # ------------------------------------------ #
# positional_embedding = positional_embedding(length, config().node_embed_size)
# order = np.argsort(np.sum(positional_embedding.detach().cpu().numpy(), axis=1))
# # ------------------------------------------ #
# X_samples_ordered = []
# for i in range(X_samples.shape[0]):
#     X_samples_ordered.append(normalize_coordinates(
#         compute_reordered_coordinate(X_samples.cpu().numpy()[i,:,:], order)))
# X_samples_ordered = np.array(X_samples_ordered)
# # ------------------------------------------ #
# H_samples_ordered = []
# for i in range(H_samples.shape[0]):
#     H_samples_ordered.append(Numpy_normalize(
#         compute_reordered_coordinate(H_samples.cpu().numpy()[i,:,:], order)))
# H_samples_ordered = np.array(H_samples_ordered)
# # ========================================== #
# #            Perfomance evaluation           #
# # ========================================== #
# real_coordinates = []
# real_coordinates_reshaped = []
# aligned_real, aligned_gen = [], []
# Frechet_dist, density_coverage = [], []
# # ------------------------------------------ #
# real_features = []
# Frechet_dist_features = []
# real_features_reshaped = []
# density_coverage_features = []
# # ------------------------------------------ #
# with open('Dataset/Test_32.pkl', 'rb') as file:
#     Test_dataset = pickle.load(file)
# # =============================================
# for i in range(len(DataLoader(CustomDataset(Test_dataset), 
#                        num_real_instances, shuffle=True))):
#     # -----------------------------------------
#     real_coordinates.append(next(iter(DataLoader(CustomDataset(Test_dataset), 
#                            num_real_instances, shuffle=True)))[0][:,:length,:])
#     real_coordinates_reshaped.append(
#                            real_coordinates[-1].reshape(-1, real_coordinates[-1].shape[-1]))
#     # -----------------------------------------
#     aligned_real_, aligned_gen_ = align_data_with_ground_truth(
#                                                             real_coordinates[-1].cpu().numpy(), 
#                                                             X_samples_ordered)
#     aligned_real.append(aligned_real_)
#     aligned_gen.append(aligned_gen_)
#     # -----------------------------------------
#     Frechet_dist.append(Frechet_distance(aligned_real[-1], aligned_gen[-1]))
#     # -----------------------------------------
#     density_coverage.append(       
#                             compute_prdc(
#                             real_instances = aligned_real[-1],
#                             generated_instances = aligned_gen[-1], 
#                             nearest_k=nearest_k))
#     # =========================================
#     real_features.append(next(iter(DataLoader(CustomDataset(Test_dataset), 
#                            num_real_instances, shuffle=True)))[1][:,:length,:])
#     real_features_reshaped.append(real_features[-1].reshape(-1, real_features[-1].shape[-1]))
#     # -----------------------------------------
#     H_samples_ordered_reshaped = H_samples_ordered.reshape(-1, H_samples_ordered.shape[-1])
#     Frechet_dist_features.append(
#                             Frechet_distance(real_features_reshaped[-1], 
#                             H_samples_ordered_reshaped))
#     # -----------------------------------------
#     real_features_reshaped.append(
#                                 real_features[-1].reshape(real_features[-1].shape[0], 
#                                 real_features[-1].shape[1]*real_features[-1].shape[-1]))
#     H_samples_ordered_reshaped = H_samples_ordered.reshape(H_samples_ordered.shape[0], 
#                                                    H_samples_ordered.shape[1]*H_samples_ordered.shape[-1])
#     density_coverage_features.append(
#                                     compute_prdc(
#                                     real_instances = real_features_reshaped[-1],
#                                     generated_instances = H_samples_ordered_reshaped, 
#                                     nearest_k=nearest_k))
# # =============================================
# density_coordinates = np.mean([x['density'] for x in density_coverage])
# coverage_coordinates = np.mean([x['coverage'] for x in density_coverage])
# density_features = np.mean([x['density'] for x in density_coverage_features])
# coverage_features = np.mean([x['coverage'] for x in density_coverage_features])
# Frechet_Coordinates = np.mean(Frechet_dist)
# Frechet_features = np.mean(Frechet_dist_features)
# # =============================================
# 
# =============================================================================



