""" ########## Processing Protein backbone coordinates and distance matrices ########## """
import os
import torch
import pickle
import random
import pandas as pd 
from Functions import Functions, random_rotation, normalize_coordinates
# ========================================= #
padding = Functions("/Dataset/").padding
encode_CT = Functions("/Dataset/").encode_CT
# ========================================= #
class Data_Processing():
    def __init__(
                self, pad_length = 32, Data_Aug_Folds = 2, 
                 alpha_C_dir = '/Dataset/PDB_alpha_C_',
                 seq_dir = 'Dataset/AA_Seq_main.csv'
                 ):
        self.seq_dir = seq_dir
        self.pad_length = pad_length
        self.Data_Aug_Folds = Data_Aug_Folds
        self.alpha_C_dir = os.getcwd() + alpha_C_dir
        self.files = os.listdir(self.alpha_C_dir)
    # ========================================= #
    #               Backbone_Coordinates        #
    # ========================================= #
    def Backbone_Coordinates(self, Directory, files):
        # files = os.listdir(Directory)
        Coordinates = {}
        Protein_backbone = {}
        Dist_Protein_backbone = {}
        for file in files: 
            coordinate = pd.read_excel(Directory +'/'+file, 
                            names=['X_coordinate', 'Y_coordinate', 'Z_coordinate']).to_numpy()
                # -------------------------------------------- #
            if coordinate.shape[0] > self.pad_length:
                cut_coordinate = coordinate[:self.pad_length, :]
                cut_coordinate = normalize_coordinates(cut_coordinate)
                Coordinates[file.replace('.xlsx','')] = cut_coordinate
                # -------------------------------------------- #
                Protein_backbone[file.replace('.xlsx','')] = cut_coordinate
                # -------------------------------------------- #
            else:
                coordinates_ = padding(self.pad_length, coordinate)
                coordinates_ = normalize_coordinates(coordinates_)
                Coordinates[file.replace('.xlsx','')] = coordinates_
                # -------------------------------------------- #
                Protein_backbone[file.replace('.xlsx','')] = Coordinates[file.replace('.xlsx','')][:,:]
        with open('Dataset/Protein_Backbone_32.pkl', 'wb') as file:
            pickle.dump(Protein_backbone, file)
    # ========================================= #
    #         Encoded Protein Backbone Seq      #
    # ========================================= #
    def Encoded_Backbone_Seq(self, Directory):
        # Directory = 'Dataset/AA_Seq_main.csv'
        AA_dataset = pd.read_csv(Directory)
        Encoded_AA = encode_CT(self.pad_length, AA_dataset)
        with open('Dataset/Backbone_Features_32.pkl', 'wb') as file:
            pickle.dump(Encoded_AA, file)
    # ========================================
    def filter_dict_by_keys(self, original_dict, keys_to_keep, seq_len):
        return {key: original_dict[key][:seq_len,:] for key in keys_to_keep if key in original_dict}
    # ========================================= #
    #             Processing Datasets           #
    # ========================================= #
    def Processing(self):
        Load_Data = self.Backbone_Coordinates(self.pad_length, self.alpha_C_dir, self.files)    
        Encoded_features = self.Encoded_Backbone_Seq(self.pad_length, self.seq_dir)
        # ==========================================
        # protein backbone Coordinates -> X
        with open('Dataset/Protein_Backbone_32.pkl', 'rb') as file:
            Backbone = pickle.load(file)
        file.close()
        # ==========================================
        # Backbone Features -> h
        with open('Dataset/Backbone_Features_32_.pkl', 'rb') as file:
            Features = pickle.load(file)
        # ==========================================
        IDs = torch.load('Dataset/Proteins_PDB_ID.pt')
        # ==========================================
        Match_keys = [item for item in IDs if item in list(Features.keys())]
        Backbone_Dict = self.filter_dict_by_keys(Backbone, Match_keys, self.pad_length)
        Features_Dict = self.filter_dict_by_keys(Features, Match_keys, self.pad_length)
        # ==========================================
        # Saving Backbone Coordinates -> X (Matched datasets)
        with open('Dataset/Backbone_Dict_32.pkl', 'wb') as file:
            pickle.dump(Backbone_Dict, file)
        # ==========================================
        # Saving Backbone Features -> h (Matched datasets)
        with open('Dataset/Backbone_Features_Dict_32.pkl', 'wb') as file:
            pickle.dump(Features_Dict, file)
    # ==============================================
    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    # ========================================= #
    #             Augmenting Datasets           #
    # ========================================= #
    def Data_Augmentation(self):
        backbone_file = 'Dataset/Backbone_Dict_32.pkl'
        features_file = 'Dataset/Backbone_Features_Dict_32.pkl'
        if not os.path.exists(backbone_file) and not os.path.exists(features_file):
            self.Processing()
        Backbone_Dict = self.load_pickle(backbone_file)
        Features_Dict = self.load_pickle(features_file)
        # --------------------------------------------------
        DATASET = []
        for key, backbone in Backbone_Dict.items():
            normalized_coordinates = normalize_coordinates(backbone)
            features = Features_Dict[key]
            DATASET.append([normalized_coordinates, features])
            for _ in range(self.Data_Aug_Folds):
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
# ==================================================
if __name__ == '__main__':
    Data_Processing(pad_length = 32, Data_Aug_Folds = 20).Data_Augmentation()

