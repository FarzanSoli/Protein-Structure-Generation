import os
import math
import torch
import zipfile
import numpy as np
import gzip, shutil
import pandas as pd 
from torch import nn
import networkx as nx
from Config import config
from AA_features import features
from itertools import combinations
from sklearn.decomposition import PCA
from Sampling import Diffusion_Process
# ========================================= #
class Functions():
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
    # ========================================= #
    #                Fix AA sequenc             #
    # ========================================= #
    def missing_seq(self, Dataset):
        df = Dataset.loc[Dataset['Sequence'].str.contains('X') & 
                           Dataset['Sequence'].str.startswith('X') &
                           Dataset['Sequence'].str.endswith('X')]
        for i in ['A','C','D','E','F','G','H','I','K','L',
                  'M','N','P','Q','R','S','T','V','W','Y']:
            kn_seq = df.loc[Dataset['Sequence'].str.contains(i)]
        result = pd.concat([df, kn_seq]).drop_duplicates(keep=False)
        return result
    # ========================================= #
    #                 Read Dataset              #
    # ========================================= #
    def Dataset_Reader(self, dataset):
        Dataset = pd.concat([pd.read_csv('Dataset/'+dataset), 
                             self.missing_seq(pd.read_csv(
                            'Dataset/'+dataset))]).drop_duplicates(keep=False)
        return Dataset
    # ========================================= #
    #                   Padding                 #
    # ========================================= #
    def padding(self, pad_len, matrix):
        mat = np.zeros((pad_len, 3))
        mat[:matrix.shape[0], :matrix.shape[-1]] = matrix
        return mat
    # ========================================= #
    #                  Unzip files              #
    # ========================================= #
    def unzip(self, directory, file):
        # file = 'PDB alpha-C.zip'
        # directory = "/Dataset/"
        with zipfile.ZipFile(os.getcwd()+directory+file, 'r') as zip_ref:
            zip_ref.extractall(os.getcwd()+directory)
    # ========================================= #
    #                 Extract gzip              #
    # ========================================= #
    def gz_extract(self):
        extension = ".gz"
        os.chdir(self.directory)
        for item in os.listdir(self.directory): # loop through items in dir
          if item.endswith(extension): # check for ".gz" extension
              gz_name = os.path.abspath(item) # get full path of files
              # get file name for file within -> removes '.cif'
              file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] 
              with gzip.open(gz_name, "rb") as f_in, open(file_name, "wb") as f_out:
                  # Copy the contents of source file to destination file
                  shutil.copyfileobj(f_in, f_out)
              os.remove(gz_name) # delete zipped file
    # ========================================= #
    #               Encoding Amio acids         #
    # ========================================= #
    def Normalize_AA(self):
        Norm_props = {}
        for key, value in features().Props.items():
            Norm_props[key] = (value-np.min(value))/(np.max(value)-np.min(value))
        # ------------------------------------- #
        Norm_AAs = {}
        for symb, ind in features().AA_dict.items():
            props = []
            for key, value in Norm_props.items():
                props.append(value[ind])
            Norm_AAs[symb] = props
        return Norm_props, Norm_AAs
    # ========================================= #
    #               Amino acid Encoding         #
    # ========================================= #
    def encode_CT(self, Pad_Length, dataframe):
        Encoded_AA = {}
        for index, row in dataframe.iterrows():
            Encoded_AA[row['PDB_ID']] = np.array(
                [self.Normalize_AA()[1][c.upper()] for c in row['Sequence']])
        # ------------------------------------- #
        encoded_AA = np.zeros((Pad_Length, len(features().AA_prop_keys)))
        Encoded_AA_padded = {}
        for key, value in Encoded_AA.items():
            if value.shape[0] > Pad_Length:
                Encoded_AA_padded[key] = value[:Pad_Length,:]
            else: 
                padding = np.zeros((Pad_Length, value.shape[1]))
                padded_value = np.vstack((value, padding))
                Encoded_AA_padded[key] = padded_value
        return Encoded_AA_padded
    # ========================================= #
    #                 READ Fasta file           #
    # ========================================= #
    def read_fasta(self, fasta_file, comment="#"):
        # with gzip.open(fasta_file,"r") as f:
        with open(fasta_file, "r") as file:
            id_ = None
            seq_id = []
            sequence = []
            sequences = []
            # loop through each line in the file
            for line in file:
                # If the line starts with a ">" character, 
                # it is a new sequence identifier
                if line.startswith(">"):
                    # If this is not the first sequence, print the previous one
                    if id_ is not None:
                        seq_id.append(id_)
                        sequences.append(''.join(sequence))
                    # Get the new sequence identifier and reset the sequence variable
                    id_ = line.strip()[1:]
                    sequence = []
                # Otherwise, it is part of the sequence, 
                # so append it to the sequence variable
                else:
                    sequence.append(line.upper())
            if id_ is not None:
                seq_id.append(id_)
                sequences.append(''.join(sequence))
            return list(zip(seq_id, sequences))
# ========================================= #
#                Numpy Normalize            #
# ========================================= #
def Numpy_normalize(points):
    # Calculate the minimum and maximum values along the specified dimension
    min_vals = np.min(points, axis=0, keepdims=True)
    max_vals = np.max(points, axis=0, keepdims=True)
    
    # Normalize the tensor along the specified dimension
    normalized_points = (points - min_vals) / (max_vals - min_vals + 1e-8)  
    return normalized_points
# ========================================= #
#               Tensor Normalize            #
# ========================================= #
def Tensor_normalize(tensor, dim=0):
    # Calculate the minimum and maximum values along the specified dimension
    min_vals, _ = torch.min(tensor, dim=dim, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=dim, keepdim=True)
    # Normalize the tensor along the specified dimension
    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals + 1e-8) 
    return normalized_tensor
# ========================================= #
#             Coordinate Normalize          #
# ========================================= #
def normalize_coordinates(coordinates):
    center_of_gravity = np.mean(coordinates, axis=0, keepdims=True)
    centered_coordinates = coordinates - center_of_gravity
    norms = np.linalg.norm(centered_coordinates, axis=-1, keepdims=True)
    normalized_coordinates = centered_coordinates / norms

    return normalized_coordinates
# ========================================= #
#             Coordinate Normalize          #
# ========================================= #
def torch_normalize_coordinates(coordinate):
    norms = torch.linalg.norm(coordinate, axis=-1, keepdim=True)
    epsilon = 1e-8
    norms = torch.where(norms == 0, torch.tensor(epsilon, device=norms.device), norms)
    normalized_coordinates = coordinate / norms
    return normalized_coordinates
# ========================================= #
#            distance mat Normalize         #
# ========================================= #
def normalize_distance_matrix(dist_matrix):
    """
    Normalize a distance matrix so that distances fall within the range [0, 1].
    """
    min_dist = np.min(dist_matrix)
    max_dist = np.max(dist_matrix)
    normalized_matrix = (dist_matrix - min_dist) / (max_dist - min_dist)
    return normalized_matrix
# ========================================= #
#              Align Coordinates            #
# ========================================= #
def compute_reordered_coordinate(coordinate, order):
    reordered_coordinate = coordinate[order]
    return reordered_coordinate
# =========================================
def Frechet_distance(Flatten_Real, Flatten_Generated):
    # Ensure inputs are numpy arrays
    Flatten_Real = np.array(Flatten_Real)
    Flatten_Generated = np.array(Flatten_Generated)
    
    # Compute squared Euclidean distance
    Sqrd_Euclid_Dist = np.sum((Flatten_Real.mean(axis=0) - Flatten_Generated.mean(axis=0)) ** 2)
    
    # Compute covariance matrices
    Covar_Real = np.cov(Flatten_Real.T)
    Covar_Gen = np.cov(Flatten_Generated.T)
    
    # Compute square root of covariance matrices using eigenvalue decomposition
    _, sqrtm_Covar_Gen = np.linalg.eigh(Covar_Gen)
    _, sqrtm_Covar_Real = np.linalg.eigh(Covar_Real)
    
    # Compute trace
    Trace_covar = np.trace(np.abs(Covar_Gen + Covar_Real - 2 * np.dot(sqrtm_Covar_Gen, sqrtm_Covar_Real)))
    
    # Compute Frechet distance
    frech_dist = Sqrd_Euclid_Dist + Trace_covar
    return frech_dist
# =========================================
# Function to compute the inertia tensor
def compute_inertia_tensor(points):
    centered_points = points - np.mean(points, axis=(0, 1), keepdims=True)
    inertia_tensor = np.einsum('ijk,ijl->kl', centered_points, centered_points)
    return inertia_tensor
# =========================================
# Function to extract principal axes from inertia tensor
def extract_principal_axes(inertia_tensor):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    principal_axes = eigenvectors[:, sorted_indices]
    return principal_axes
# =========================================
# Function to align data using PCA and principal axes
def align_data_with_ground_truth(real_data, gen_data):
    # Compute inertia tensor and principal axes for real data
    inertia_real = compute_inertia_tensor(real_data)
    principal_axes_real = extract_principal_axes(inertia_real)
    # Fit PCA to the aligned real data
    real_data_reshaped = real_data.reshape(-1, real_data.shape[-1])
    pca = PCA(n_components=3)
    pca.fit(real_data_reshaped.dot(principal_axes_real))
    aligned_real = pca.transform(real_data_reshaped)
    # Transform generated data using the same PCA
    generated_data_reshaped = gen_data.reshape(-1, gen_data.shape[-1])
    aligned_gen = pca.transform(generated_data_reshaped)
    return normalize_coordinates(aligned_real), normalize_coordinates(aligned_gen)
# ========================================= 
def generate_laplacian_noise(shape, loc=0, scale=1):
    # Create a Laplace distribution object
    laplace_dist = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
    # Sample noise from the Laplace distribution
    noise = laplace_dist.sample(shape)
    return noise
# ========================================= 
def generate_uniform_noise(shape, low=0, high=1):
    # Generate uniform noise tensor
    noise = torch.rand(shape) * (high - low) + low
    return noise
# ========================================= 
def construct_graph(coords, threshold=0.1):
    """
    Construct a graph from 3D coordinates.
    
    Parameters:
    coords (numpy.ndarray): 3D coordinates of shape (N, 3) for a protein sequence
    threshold (float): Threshold distance for edge connectivity
    
    Returns:
    networkx.Graph: The constructed graph
    """
    G = nx.Graph()
    
    # Add nodes to the graph based on the coordinates
    for i, coord in enumerate(coords):
        G.add_node(i, pos=tuple(coord))
    
    # Add edges based on the distance threshold
    for u, v in combinations(G.nodes(), 2):
        dist = np.linalg.norm(coords[u] - coords[v])
        if dist < threshold:
            G.add_edge(u, v)
    return G
# ========================================= 
def graph_laplacian_spectrum(coords, threshold=0.1):
    """
    Compute the Laplacian spectrum of a graph.
    
    Parameters:
    graph (networkx.Graph): Input graph
    
    Returns:
    numpy.ndarray: The eigenvalues of the graph Laplacian
    """
    # Construct graph
    G = construct_graph(coords, threshold)
    # Compute Laplacian matrix
    L = nx.laplacian_matrix(G).astype(float)
    # Compute Laplacian spectrum (eigenvalues)
    eigenvalues = np.linalg.eigvals(L.todense())
    return np.sort(eigenvalues)
# ========================================= 
def compare_laplacian_spectra(coords_set1, coords_set2, threshold=0.1):
    """
    Compare the Laplacian spectra of two sets of 3D coordinates.

    Parameters:
    coords_set1 (numpy.ndarray): The first set of coordinates of shape (N, 32, 3)
    coords_set2 (numpy.ndarray): The second set of coordinates of shape (N, 32, 3)
    threshold (float): Threshold distance for edge connectivity

    Returns:
    float: A measure of similarity between the average spectra of the two sets of coordinates
    """
    spectra1 = [graph_laplacian_spectrum(coords, threshold) for coords in coords_set1]
    spectra2 = [graph_laplacian_spectrum(coords, threshold) for coords in coords_set2]
    
    mean_spectrum1 = np.mean(spectra1, axis=0)
    mean_spectrum2 = np.mean(spectra2, axis=0)
    
    return np.linalg.norm(mean_spectrum1 - mean_spectrum2)

# =========================================
class RandomRotation(nn.Module):
    # -------------------------------------------
    def __init__(self):
      super().__init__()
      self.angle_range = (-math.pi, math.pi)
    # -------------------------------------------
    def forward(self, points):
      # Sample random rotation angles for each axis
      rand = np.random.rand(3)
      theta = rand * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]
      # -------------------------------------------
      # Rotation matrix for X-axis rotation      
      Rx = torch.Tensor([[np.cos(theta[0]), -np.sin(theta[0]), 0],
                       [np.sin(theta[0]),  np.cos(theta[0]), 0],
                       [0,                   0,             1]])
      # -------------------------------------------    
      # Rotation matrix for Y-axis rotation
      Ry = torch.Tensor([[np.cos(theta[1]), 0,  np.sin(theta[1])],
                        [0,                   1,               0],
                        [-np.sin(theta[1]), 0,  np.cos(theta[1])]])
      # -------------------------------------------
      # Rotation matrix for Z-axis rotation
      Rz = torch.Tensor([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                        [np.sin(theta[2]),  np.cos(theta[2]), 0],
                        [0,              0,                   1]])
      # -------------------------------------------
      # Combine rotations for all axes (assuming intrinsic rotations)
      rotation_matrix = np.einsum('ijk,jk->ik', np.stack([Rx, Ry, Rz]), np.eye(3))
      # Rotate points
      return np.einsum('bi,ij->bj', points, rotation_matrix)
# =========================================
# Rotation data augmntation
def random_rotation(x):
    n_nodes, n_dims = x.shape
    angle_range = 2 * np.pi
    if n_dims == 2:
        theta = np.random.rand() * angle_range - np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        x = x.T
        x = np.dot(R, x)
        x = x.T
    elif n_dims == 3:
        # Build Rx
        theta = np.random.rand() * angle_range - np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)
        Rx = np.array([[1, 0, 0],
                       [0, cos, -sin],
                       [0, sin, cos]])
        # Build Ry
        theta = np.random.rand() * angle_range - np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)
        Ry = np.array([[cos, 0, sin],
                       [0, 1, 0],
                       [-sin, 0, cos]])
        # Build Rz
        theta = np.random.rand() * angle_range - np.pi
        cos = np.cos(theta)
        sin = np.sin(theta)
        Rz = np.array([[cos, -sin, 0],
                       [sin, cos, 0],
                       [0, 0, 1]])
        # Apply rotations
        x = x.T
        x = np.dot(Rx, x)
        x = np.dot(Ry, x)
        x = np.dot(Rz, x)
        x = x.T
    else:
        raise Exception("Not implemented Error")
    return x
# ==============================================
# Create DataLoader with shuffle enabled
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
# ==============================================
def dynamic_weighting(loss1, loss2, log_var_x, log_var_f):
    precision1 = torch.exp(-log_var_x)
    precision2 = torch.exp(-log_var_f)
    weighted_loss = precision1 * loss1 + precision2 * loss2 + log_var_x + log_var_f
    return weighted_loss
# ==============================================
def Sampling(model, Samples, eta, device, nearest_k = 5, only_final = True):
    if eta == 1:
        method = 'DDPM'
    elif eta == 0:
        method = 'DDIM'
    # ----------------------------------------------- #
    num_real_instances = Samples
    # ----------------------------------------------- #
    features_shape = (Samples, config().num_residues, config().num_features)
    sample_features = torch.randn(*features_shape).to(device = device)
    # ----------------------------------------------- #
    coordinates_shape = (Samples, config().num_residues, 3)
    sample_coordinate = torch.randn(*coordinates_shape).to(device = device)
    # ----------------------------------------------- #
    Diffsion = Diffusion_Process(model, coordinates_shape, features_shape, eta = eta)
    X_samples = Diffsion.Sampling(sample_coordinate = sample_coordinate.to(device = device), 
                           sample_features = sample_features.to(device = device),
                           only_final = only_final)[0]
    # -----------------------------------------
    x_dir = 'Dataset/'+method+'/x_sample.pt'
    torch.save(X_samples, x_dir)
    # ==========================================
    H_samples = Diffsion.Sampling(sample_coordinate = sample_coordinate.to(device = device), 
                                   sample_features = sample_features.to(device = device),
                                   only_final = only_final)[1]
    # -----------------------------------------
    h_dir = 'Dataset/'+method+'/h_sample.pt'
    torch.save(H_samples, h_dir)
    return X_samples, H_samples
