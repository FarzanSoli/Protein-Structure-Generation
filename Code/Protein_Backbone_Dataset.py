""" ########## Processing Protein backbone coordinates and distance matrices ########## """
import os
import wget
import copy
import torch
import pickle
import random
import zipfile
import numpy as np
import pandas as pd 
from Functions import Functions, random_rotation, normalize_coordinates
# ========================================= #
padding = Functions("/Dataset/").padding
encode_CT = Functions("/Dataset/").encode_CT
# ========================================= #
#               Backbone_Coordinates        #
# ========================================= #
def Backbone_Coordinates(Pad_Length, Directory, files):
    # Pad_Length = 32
    # Directory = os.getcwd() +'/Dataset/PDB_alpha_C'
    # files = os.listdir(Directory)
    Coordinates = {}
    Protein_backbone = {}
    Dist_Protein_backbone = {}
    for file in files: 
        coordinate = pd.read_excel(Directory +'/'+file, 
                        names=['X_coordinate', 'Y_coordinate', 'Z_coordinate']).to_numpy()
            # -------------------------------------------- #
        if coordinate.shape[0] > Pad_Length:
            cut_coordinate = coordinate[:Pad_Length, :]
            cut_coordinate = normalize_coordinates(cut_coordinate)
            Coordinates[file.replace('.xlsx','')] = cut_coordinate
            # -------------------------------------------- #
            Protein_backbone[file.replace('.xlsx','')] = cut_coordinate
            # -------------------------------------------- #
        else:
            coordinates_ = padding(Pad_Length, coordinate)
            coordinates_ = normalize_coordinates(coordinates_)
            Coordinates[file.replace('.xlsx','')] = coordinates_
            # -------------------------------------------- #
            Protein_backbone[file.replace('.xlsx','')] = Coordinates[file.replace('.xlsx','')][:,:]
    with open('Dataset/Protein_Backbone_32.pkl', 'wb') as file:
        pickle.dump(Protein_backbone, file)
# ========================================= #
#         Encoded Protein Backbone Seq      #
# ========================================= #
def Encoded_Backbone_Seq(Pad_Length, Directory):
    # Directory = 'Dataset/AA_Seq_main.csv'
    AA_dataset = pd.read_csv(Directory)
    Encoded_AA = encode_CT(Pad_Length, AA_dataset)
    with open('Dataset/Backbone_Features_32.pkl', 'wb') as file:
        pickle.dump(Encoded_AA, file)
# ========================================= #
#           Padding Protein backbone        #
# ========================================= #
def Padding_Protein_Backbone(Pad_Length, Encoded_AA):
    # Pad_Length = 32
    Backbone_Seq = {}
    for key, value in Encoded_AA.items():
        Backbone_Seq[key] = np.zeros((Pad_Length, value.shape[-1]))
        Backbone_Seq[key][:value[:Pad_Length].shape[0],
                       :value[:Pad_Length].shape[-1]] = value[:Pad_Length]
    # -----------------------------------------------------
    with open('Dataset/Padded_Backbone_Seq_32.pkl', 'wb') as file:
        pickle.dump(Backbone_Seq, file)
# ==========================================
Pad_Length = 32
data_dir = os.getcwd() +'/Dataset/PDB_alpha_C_'
files = os.listdir(data_dir)
Load_Data = Backbone_Coordinates(Pad_Length, data_dir, files)

Directory = 'Dataset/AA_Seq_main.csv'
Encoded_features = Encoded_Backbone_Seq(Pad_Length, Directory)
# ==========================================
# protein backbone Coordinates -> X
with open('Dataset/Protein_Backbone_32.pkl', 'rb') as file:
    Backbone = pickle.load(file)
file.close()
# ==========================================
# Backbone Features -> h
with open('Dataset/Backbone_Features_32_.pkl', 'rb') as file:
    Features = pickle.load(file)
file.close()
IDs = torch.load('Dataset/Proteins_PDB_ID.pt')
# ==========================================
Match_keys = [item for item in IDs if item in list(Features.keys())]
def filter_dict_by_keys(original_dict, keys_to_keep, seq_len):
    return {key: original_dict[key][:seq_len,:] for key in keys_to_keep if key in original_dict}
Backbone_Dict = filter_dict_by_keys(Backbone, Match_keys,32)
Features_Dict = filter_dict_by_keys(Features, Match_keys,32)
# ==========================================
# Saving Backbone Coordinates -> X (Matched datasets)
with open('Dataset/Backbone_Dict_32.pkl', 'wb') as file:
    pickle.dump(Backbone_Dict, file)
file.close()
# ==========================================
# Saving Backbone Features -> h (Matched datasets)
with open('Dataset/Backbone_Features_Dict_32.pkl', 'wb') as file:
    pickle.dump(Features_Dict, file)
file.close()
# ==============================================
# =============================================================================
# # Backbone Coordinates -> X
# with open('Dataset/Backbone_Dict_32.pkl', 'rb') as file:
#     Backbone_Dict = pickle.load(file)
# # ==============================================
# # Backbone Features -> h
# with open('Dataset/Backbone_Features_Dict_32.pkl', 'rb') as file:
#     Features_Dict = pickle.load(file)
# =============================================================================
# ==============================================
def Data_Augmentation(N = 2):
    DATASET = []
    for key, backbone in Backbone_Dict.items():
        normalized_coordinates = normalize_coordinates(backbone)
        features = Features_Dict[key]
        DATASET.append([normalized_coordinates, features])
        for _ in range(N):
            rotated_coordinates = normalize_coordinates(random_rotation(backbone))
            DATASET.append([rotated_coordinates, features])
    # ---------------------------------------
    # Define the sizes of each split
    train_size = int(0.8 * len(DATASET))
    val_size = int(0.1 * len(DATASET))
    test_size = len(DATASET) - train_size - val_size
    # ---------------------------------------
    train_dataset = random.sample(DATASET, train_size)
    val_dataset = random.sample(DATASET, val_size)
    test_dataset = random.sample(DATASET, test_size)
    # ---------------------------------------
    datasets = {
        'Dataset/Train_32.pkl': train_dataset,
        'Dataset/Validation_32.pkl': val_dataset,
        'Dataset/Test_32.pkl': test_dataset
        }
    for file_path, data in datasets.items():
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

