import torch
from EGNN import *
import torch.nn as nn
from Config import config
import torch.nn.functional as F
# ========================================== #
""" ############### Training EGNN ############### """
class Denoizer(nn.Module):
    # ==========================================
    def __init__(self):
        # ----------------------------------
        super(Denoizer, self).__init__()
        context_node_nf = 0
        self.device = config().device
        self.h_embed_size = config().h_embed_size
        self.num_features = config().num_features
        self.edge_embed_size = config().edge_embed_size
        # ----------------------------------
        self.layers = []
        for i in range(config().num_layers):
            layer = []
            self.egnn = EGNN(in_node_nf=self.num_features, 
                    hidden_nf=self.h_embed_size, out_node_nf=self.num_features, 
                    in_edge_nf=self.edge_embed_size, normalize=True).to(self.device)
            layer.append(self.egnn)
            layer.append(nn.LayerNorm(self.num_features+self.h_embed_size))
            self.layers.append(layer)
        self.layers_pytorch = nn.ModuleList([l for sublist in self.layers for l in sublist])
    # --------------------------------------------------- #
    def initialize(self):
        init.xavier_uniform_(self.layer.weight)
        init.zeros_(self.layer.bias)
    # ==========================================
    def forward(self, coordinates, features, idx):
        # ----------------------------------
        # corrupted bounding box coordinates of the protein. 
        # [B, N, 3]
        bb_pos = coordinates.type(torch.float32)
        curr_pos = bb_pos.clone() 
        # [B, N, feat_size]
        bb_feat = features.type(torch.float32)
        curr_feat = bb_feat.clone() 
        # ----------------------------------
        # Edge positional embeddings
        distances,_ = coord2diff(coordinates)
        sin_embedding = Sinusoidal_Embedding()
        edge_attr = sin_embedding(distances)
        # ----------------------------------
        # Node positional embeddings
        # Shape = Batch_size
        embed_N, embed_T = combined_node_embeddings(coordinates.shape[0], 
                                                    coordinates.shape[1], 
                                                    self.h_embed_size, idx)
        node_attr = embed_N + embed_T
        node_attr = node_attr.type(torch.float32)
        # curr_feat = torch.cat([curr_feat, node_attr], dim=-1)
        # ----------------------------------
        for layer in self.layers:
            egnn, norm = layer
        curr_feat, curr_pos = egnn(curr_feat, curr_pos, 
                                        edge_attr, node_attr, mask=None)
        # ----------------------------------
        eps_theta_x = curr_pos - bb_pos
        eps_theta_x = eps_theta_x.to(device=self.device)
        
        eps_theta_f = (curr_feat - bb_feat)
        eps_theta_f = eps_theta_f.to(device = self.device)
        # ----------------------------------
        eps_theta_x = eps_theta_x.reshape(bb_pos.shape)
        eps_theta_f = eps_theta_f.reshape(bb_feat.shape)
        return eps_theta_x, eps_theta_f 
    
""" ############### Noise Prediction ############### """
class Noise_Pred(nn.Module):
    def __init__(self):
        super().__init__()
        self.Denoizer = Denoizer()
        self.device = config().device
        self.alpha_bars = config().alpha_bars
        self.to(device = self.device)
    # ==================================================== # 
    #                     Forward Diffusion                #
    # ==================================================== #
    def loss_fn(self, coordinates, features, idx=None):
        # Call the forward method to get predictions and targets
        epsilon_theta_x, epsilon_theta_f, epsilon_t_x, epsilon_t_f = self(coordinates, 
                                                        features, idx, get_target=True)
        # Define MSE loss functions
        l_x = nn.MSELoss()
        l_f = nn.MSELoss()
        # Calculate MSE losses
        loss_x = l_x(epsilon_theta_x, epsilon_t_x)
        loss_f = l_f(epsilon_theta_f, epsilon_t_f)
        return loss_x.to(self.device), loss_f.to(self.device)
    # --------------------------------------------------- #
    def forward(self, x, f, idx=None, get_target=False):
        # (training phase)
        if idx == None:
            idx = torch.randint(0, len(self.alpha_bars),(x.size(0),)).to(device = self.device)
            alpha_bar_t = self.alpha_bars[idx][:, None, None]
            epsilon_t_x = torch.randn(size=x.size(), device=x.device)
            epsilon_t_f = torch.randn(size=f.size(), device=f.device)
            # noisy data (Eq. 4)
            x_t = torch.sqrt(alpha_bar_t)*x + torch.sqrt(1-alpha_bar_t)*epsilon_t_x
            f_t = torch.sqrt(alpha_bar_t)*f + torch.sqrt(1-alpha_bar_t)*epsilon_t_f
        else: 
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device=self.device).long()
            x_t = x
            f_t = f
        epsilon_theta_x, epsilon_theta_f = self.Denoizer(x_t, f_t, idx)
        return (epsilon_theta_x, epsilon_theta_f, epsilon_t_x, epsilon_t_f) if get_target else (epsilon_theta_x, epsilon_theta_f)


