""" ################## Sampling (Generation) ################## """  
import torch
import pickle
import numpy as np
from Config import config
from EGNN import positional_embedding
from torch.utils.data import DataLoader
from Training_Model import Training_Model
from Density_Coverage import compute_prdc
from Functions import Frechet_distance, Sampling
from Functions import Numpy_normalize, normalize_coordinates, CustomDataset
from Functions import align_data_with_ground_truth, compute_reordered_coordinate
# ============================================
device = torch.device('cuda:0')
model = Training_Model().train()
# eta = 0 --> DDIM 
# eta = 1 --> DDPM 
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
    density_coverage_features.append(compute_prdc(
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
