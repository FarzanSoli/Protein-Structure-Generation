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
checkpoint_dir = os.path.join(os.getcwd(),'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
# ========================================== #
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
# ============================================== #
class Training_Model:
    # ========================================== #
    def __init__(self, 
                 num_epochs,
                 Data_Aug_Folds,
                 lr,
                 batch_size,
                 ):
        super(Training_Model, self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = config().device
        self.length = config().num_residues
        self.Data_Aug_Folds = Data_Aug_Folds
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        self.log_var_x = nn.Parameter(torch.zeros(()))
        self.log_var_f = nn.Parameter(torch.zeros(()))
        # ------------- Load Dataset ------------- #
        Data_Processing().Data_Augmentation(self.Data_Aug_Folds)
    # ========================================== #
    def train(self):
        training_data_path = 'Dataset/Train_32.pkl'
        try:            
            with open(training_data_path, 'rb') as file:
                train_data = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {training_data_path}. Please ensure the dataset is prepared.")
            
        Train_data_loader = DataLoader(CustomDataset(train_data), 
                                       self.batch_size, shuffle=True)
        # ------------ Import Denoizer ----------- #
        self.model = Noise_Pred()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
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
            save_checkpoint(self.model, optimizer, epoch, weighted_loss.item(), checkpoint_dir)
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(Train_data_loader)}')
        final_loss = total_loss / len(Train_data_loader)
        return final_loss, self.model
# ==================================================
if __name__ == "__main__":
    trainer = Training_Model(
                            num_epochs=1,
                            Data_Aug_Folds=1,
                            lr=1e-6,
                            batch_size=256
                            )
    final_loss, trained_model = trainer.train()



