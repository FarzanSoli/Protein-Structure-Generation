import os
import torch
import pickle
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
    def __init__(self, 
                 num_epochs,
                 Data_Aug_Folds,
                 device = torch.device('cuda:0'), 
                 ):
        super(Training_Model, self).__init__()
        self.device = device
        self.num_epochs = num_epochs
        self.Data_Aug_Folds = Data_Aug_Folds
        self.device = torch.device('cuda:0')
        self.length = config().num_residues
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        self.log_var_x = nn.Parameter(torch.zeros(()))
        self.log_var_f = nn.Parameter(torch.zeros(()))
        # ------------- Load Dataset ------------- #
        Data_Processing(self.Data_Aug_Folds).Data_Augmentation()
    # ========================================== #
    def train(self):
        try:            
            with open('Dataset/Train_32.pkl', 'rb') as file:
                train_data = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}. Please ensure the dataset is prepared.")
            
        Train_data_loader = DataLoader(CustomDataset(train_data), 
                                       config().batch_size, shuffle=True)
        # ------------ Import Denoizer ----------- #
        self.model = Noise_Pred()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 5e-7)
        # -------------------------------------- #
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for data in tqdm(Train_data_loader):
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
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(Train_data_loader)}')
        return self.model
# ==================================================
if __name__ == "__main__":
    trainer = Training_Model(device=torch.device('cuda:0'), num_epochs=1, Data_Aug_Folds=1)
    trained_model = trainer.train()

