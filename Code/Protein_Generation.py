import os
import torch
import pickle
import numpy as np
from EGNN import *
from tqdm import tqdm
from Config import config
from Training_Denoizer import Noise_Pred
from EGNN import positional_embedding
from torch.utils.data import DataLoader
from Density_Coverage import compute_prdc
from Data_Processing import Data_Processing
from Functions import Frechet_distance, dynamic_weighting, Sampling
from Functions import Numpy_normalize, normalize_coordinates, CustomDataset
from Functions import align_data_with_ground_truth, compute_reordered_coordinate
# ============================================
device = torch.device('cuda:0')
length = config().num_residues
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)
# ------------ Import Denoizer ----------- #
only_final = True
model = Noise_Pred()
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-7)
# ------------- Load Dataset ------------- #
if not os.path.exists('Dataset/Train_32.pkl'):
    Data_Processing(pad_length = 32, Data_Aug_Folds = 2).Data_Augmentation()
# ---------------------------------------- #
with open('Dataset/Train_32.pkl', 'rb') as file:
    train_data = pickle.load(file)
Train_data_loader = DataLoader(CustomDataset(train_data), 
                               config().batch_size, shuffle=True)
# -------------------------------------- #
log_var_x = nn.Parameter(torch.zeros(()))
log_var_f = nn.Parameter(torch.zeros(()))
# -------------------------------------- #
num_epochs = 1
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for data in tqdm(Train_data_loader):
        coordinates = data[0][:, :length, :].float().to(device)
        features = data[1][:, :length, :].float().to(device)     
        # Compute loss
        loss_x, loss_f = model.loss_fn(coordinates, features)
        loss_x = loss_x.detach()
        loss_f = loss_f.detach()
        weighted_loss = dynamic_weighting(loss_x, loss_f, log_var_x, log_var_f)       
        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()
        total_loss += weighted_loss.item() 
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(Train_data_loader)}')

""" ################## Sampling (Generation) ################## """    
X_samples, H_samples = Sampling(model, device=device, Samples = 1000, eta = 1)
# ------------------------------------------ #
#     Order based on positional embedding    #
# ------------------------------------------ #
positional_embedding = positional_embedding(length, config().node_embed_size)
order = np.argsort(np.sum(positional_embedding.detach().cpu().numpy(), axis=1))
# ------------------------------------------ #
X_samples_ordered = []
for i in range(X_samples.shape[0]):
    X_samples_ordered.append(normalize_coordinates(
        compute_reordered_coordinate(X_samples.cpu().numpy()[i,:,:], order)))
X_samples_ordered = np.array(X_samples_ordered)
# ------------------------------------------ #
H_samples_ordered = []
for i in range(H_samples.shape[0]):
    H_samples_ordered.append(Numpy_normalize(
        compute_reordered_coordinate(H_samples.cpu().numpy()[i,:,:], order)))
H_samples_ordered = np.array(H_samples_ordered)
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
for i in range(len(DataLoader(CustomDataset(Test_dataset), 
                       num_real_instances, shuffle=True))):
    # -----------------------------------------
    real_coordinates.append(next(iter(DataLoader(CustomDataset(Test_dataset), 
                           num_real_instances, shuffle=True)))[0][:,:length,:])
    real_coordinates_reshaped.append(
                           real_coordinates[-1].reshape(-1, real_coordinates[-1].shape[-1]))
    # -----------------------------------------
    aligned_real_, aligned_gen_ = align_data_with_ground_truth(
                                                            real_coordinates[-1].cpu().numpy(), 
                                                            X_samples_ordered)
    aligned_real.append(aligned_real_)
    aligned_gen.append(aligned_gen_)
    # -----------------------------------------
    Frechet_dist.append(Frechet_distance(aligned_real[-1], aligned_gen[-1]))
    # -----------------------------------------
    density_coverage.append(       
                            compute_prdc(
                            real_instances = aligned_real[-1],
                            generated_instances = aligned_gen[-1], 
                            nearest_k=nearest_k))
    # =========================================
    real_features.append(next(iter(DataLoader(CustomDataset(Test_dataset), 
                           num_real_instances, shuffle=True)))[1][:,:length,:])
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
    density_coverage_features.append(
                                    compute_prdc(
                                    real_instances = real_features_reshaped[-1],
                                    generated_instances = H_samples_ordered_reshaped, 
                                    nearest_k=nearest_k))
# =============================================
density_coordinates = np.mean([x['density'] for x in density_coverage])
coverage_coordinates = np.mean([x['coverage'] for x in density_coverage])
density_features = np.mean([x['density'] for x in density_coverage_features])
coverage_features = np.mean([x['coverage'] for x in density_coverage_features])
Frechet_Coordinates = np.mean(Frechet_dist)
Frechet_features = np.mean(Frechet_dist_features)
# =============================================




