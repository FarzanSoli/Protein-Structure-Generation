""" ########## Processing Protein backbone coordinates and distance matrices ########## """
import os
import math
import wget
import copy
import torch
import pickle
import zipfile
import subprocess
import numpy as np
import pandas as pd 
from Functions import Functions
from Functions import normalize_coordinates
# ========================================= #
padding = Functions("/dataset/").padding
encode_CT = Functions("/dataset/").encode_CT
# ========================================= #
# Define the directory path
dataset_dir = os.path.join(os.getcwd(), 'dataset', 'PDB_alpha_C')

# Check if the directory exists
if not os.path.exists(dataset_dir):
    print(f"Directory {dataset_dir} does not exist. Fetching dataset...")
    # Run Fetch_Dataset.py to get the dataset
    subprocess.run(['python', 'Fetch_Dataset.py'], check=True)

# Now proceed with Protein_Backbone_Dataset.py logic
print("Directory exists. Proceeding with Protein_Backbone_Dataset.py...")
# ========================================= #
#               Backbone_Coordinates        #
# ========================================= #
def Backbone_Coordinates(Pad_Length, Directory, files):
    # Pad_Length = 32
    # Directory = os.getcwd() +'/dataset/PDB_alpha_C'
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
    with open('dataset/Protein_Backbone_32.pkl', 'wb') as file:
        pickle.dump(Protein_backbone, file)
# ========================================= #
#         Encoded Protein Backbone Seq      #
# ========================================= #
def Encoded_Backbone_Seq(Pad_Length, Directory):
    # Directory = 'dataset/AA_Seq_main.csv'
    AA_dataset = pd.read_csv(Directory)
    Encoded_AA = encode_CT(Pad_Length, AA_dataset)
    with open('dataset/Backbone_Features_32_.pkl', 'wb') as file:
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
    with open('dataset/Padded_Backbone_Seq_32.pkl', 'wb') as file:
        pickle.dump(Backbone_Seq, file)
# ==========================================
Pad_Length = 32
data_dir = os.getcwd() +'/dataset/PDB_alpha_C'
files = os.listdir(data_dir)
Load_Data = Backbone_Coordinates(Pad_Length, data_dir, files)

Directory = 'dataset/AA_Seq_main.csv'
Encoded_features = Encoded_Backbone_Seq(Pad_Length, Directory)
# ==========================================
# protein backbone Coordinates -> X
with open('dataset/Protein_Backbone_32.pkl', 'rb') as file:
    Backbone = pickle.load(file)
file.close()
# ==========================================
# Backbone Features -> h
with open('dataset/Backbone_Features_32_.pkl', 'rb') as file:
    Features = pickle.load(file)
file.close()
IDs = torch.load('dataset/Proteins_PDB_ID.pt')
# ==========================================
Match_keys = [item for item in IDs if item in list(Features.keys())]
def filter_dict_by_keys(original_dict, keys_to_keep, seq_len):
    return {key: original_dict[key][:seq_len,:] for key in keys_to_keep if key in original_dict}
Backbone_Dict = filter_dict_by_keys(Backbone, Match_keys,32)
Features_Dict = filter_dict_by_keys(Features, Match_keys,32)
# ==========================================
# Backbone Coordinates -> X (Matched datasets)
with open('dataset/Backbone_Dict_32_.pkl', 'wb') as file:
    pickle.dump(Backbone_Dict, file)
file.close()
# ==========================================
# Backbone Features -> h (Matched datasets)
with open('dataset/Backbone_Features_Dict_32_.pkl', 'wb') as file:
    pickle.dump(Features_Dict, file)
file.close()
# ==========================================

