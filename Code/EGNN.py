import math
import torch
from torch import nn
from Config import config
# ==============================================
device=torch.device('cuda:0')
pi = torch.tensor(math.pi).to(device)
node_embed_size = config().node_embed_size
edge_embed_size = config().edge_embed_size
# ==============================================
""" ###### E(n) Equivariant Convolutional Layer ###### """
class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf,
                 edges_in_d=0, act_fn=nn.Tanh(), residual=True, attention=True,
                 normalize=False, coords_agg='mean', tanh=False, dropout_rate=0.1):
        super(E_GCL, self).__init__()
        # input_nf, output_nf, hidden_nf = hidden_nf, hidden_nf, hidden_nf
        self.tanh = tanh
        edge_coords_nf = 1
        self.epsilon = 1e-8
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.num_features = config().num_features
        output_nf = self.num_features
        input_edge = self.num_features * 2
        # --------------------------------------
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge+edge_coords_nf+edges_in_d, hidden_nf), act_fn, 
            nn.Linear(hidden_nf, hidden_nf), act_fn)
        # --------------------------------------
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf+self.num_features, hidden_nf), act_fn,
            nn.Linear(hidden_nf, output_nf))
        # --------------------------------------
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        # --------------------------------------
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        # Optional attention mechanism
        if self.attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf), act_fn,
                nn.Linear(hidden_nf, 1), nn.Sigmoid()
            )
    # ========================================= #
    #                   Edge Model              #
    # ========================================= #
    def edge_model(self, h, radial, edge_attr):
        B, N, _ = h.shape
        source = h[:, None].repeat(1, N, 1, 1)
        target = h[:, :, None].repeat(1, 1, N, 1)
        out = torch.cat([source, target, radial, edge_attr], dim=3)
        out = out.view(B * N * N, -1)
        out = self.edge_mlp(out)
        out = out.view(B, N, N, -1)
        
        if self.attention:
            attn_weights = self.attention_layer(out)
            out = out * attn_weights

        return out
    # ========================================= #
    #                   Node Model              #
    # ========================================= #
    def node_model(self, h, edge_attr, node_attr, mask):
        B, N, _ = h.shape
        agg = torch.sum(edge_attr, axis=2)
        agg = torch.cat([h, node_attr, agg], dim=-1)
        agg = agg * mask[:, :, None]
        agg = agg.view(B * N, -1)
        out = self.node_mlp(agg)
        out = out.view(B, N, -1)
        if self.residual:
            out = h + out
        return out
    # ==========================================
    def coord_model(self, coord, coord_diff, mask, edge_feat):
        B, N, D = coord.shape
        mask_2d = mask[:, :, None] * mask[:, None, :]
        coord_diff = coord_diff.view(B, N * N, D)
        edge_feat = edge_feat.view(B, N * N, -1)
        embed_edge = self.coord_mlp(edge_feat)
        trans = coord_diff * embed_edge
        trans = mask_2d[..., None] * trans.view(B, N, N, D)

        if self.coords_agg == 'sum':
            agg = torch.sum(trans, axis=2)
        elif self.coords_agg == 'mean':
            agg = torch.sum(trans, axis=2) / (torch.sum(mask_2d, axis=2, keepdim=True) + 1e-10)
        else:
            raise Exception(f'Wrong coords_agg parameter: {self.coords_agg}')
        coord = coord + agg
        return coord
    # ==========================================
    def coord2radial(self, coord):
        coord_diff = coord[:, :, None] - coord[:, None]
        radial = torch.sum(coord_diff ** 2, axis=-1).unsqueeze(-1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff
    # =============================================
    def forward(self, h, coord, edge_attr, node_attr=None, mask=None):
        if mask is None:
            mask = torch.ones(h.shape[:2]).to(device)
        coord *= mask[..., None]
        h *= mask[..., None]
        
        mask_2d = mask[:, :, None] * mask[:, None, :]
        radial, coord_diff = self.coord2radial(coord)
        radial *= mask_2d[..., None]
        coord_diff *= mask_2d[..., None]

        edge_feat = self.edge_model(h, radial, edge_attr)
        coord = self.coord_model(coord, coord_diff, mask, edge_feat)
        coord *= mask[:, :, None]
        h = self.node_model(h, edge_feat, node_attr, mask)
        h *= mask[:, :, None]
        return h, coord, edge_feat

""" ##############################################
    ###### Equivariant Graph Neural Network ######
    ############################################## """
class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                 device=config().device, act_fn=nn.Tanh(), 
                 n_layers=8, residual=True, attention=True,
                 normalize=False, tanh=False, dropout_rate=0.1):
        super(EGNN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_nf = hidden_nf
        self.num_features = config().num_features
        self.embedding_in = nn.Sequential(
            nn.Linear(in_node_nf, self.num_features),
            nn.LayerNorm(self.num_features),  # LayerNorm for input embedding
            act_fn
        )
        self.embedding_out = nn.Sequential(
            nn.Linear(self.num_features, out_node_nf),
            nn.LayerNorm(out_node_nf)  # LayerNorm for output embedding
        )

        for i in range(n_layers):
            self.add_module("gcl_%d" % i, E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, 
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, 
                attention=attention, normalize=normalize, tanh=tanh, dropout_rate=dropout_rate
            ))
        self.to(self.device)

    def forward(self, h, x, edge_attr, node_attr, mask=None):
        B, N, _ = h.shape
        h = h.view(B * N, -1)
        h = self.embedding_in(h)
        h = h.view(B, N, -1)
        for i in range(self.n_layers):
            h, x, edge_feat = self._modules["gcl_%d" % i](h, x, edge_attr=edge_attr, 
                                                          node_attr=node_attr, mask=mask)
        h = h.view(B * N, -1)
        h = self.embedding_out(h)
        h = h.view(B, N, -1)
        return h, x
# ============================================= #
#                 Nodes Embeddings              #
# ============================================= #
def positional_embedding(N, node_embed_size, device=torch.device('cuda:0')):
    res_index = torch.arange(N).float().to(device)
    K = torch.arange(node_embed_size // 2, dtype=torch.float).to(device)
    
    div_term = torch.exp(K * -(torch.log(torch.tensor(10000.0)) / (node_embed_size // 2))).to(device)
    
    pos_embedding_sin = torch.sin(res_index[:, None] * div_term).to(device)
    pos_embedding_cos = torch.cos(res_index[:, None] * div_term).to(device)
    
    pos_embedd = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedd
# ============================================= #
def combined_node_embeddings(T, N, node_embed_size, t, device=torch.device('cuda:0')):
    embedding_T = positional_embedding(T, node_embed_size).to(device)
    embedding_N = positional_embedding(N, node_embed_size).to(device)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    batch_size = t.shape[0]
    t = t.clamp(0, T-1)
    t_emb = embedding_T[t]  # Shape: [batch_size, node_embed_size]
    embed_T = t_emb.unsqueeze(1).expand(-1, N, -1)  # Shape: [batch_size, N, node_embed_size]
    embed_N = embedding_N.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, N, node_embed_size]
    
    return embed_N, embed_T
# ==============================================
""" ###### Sinusoids Embedding ###### """
class Sinusoidal_Embedding(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        # ---------------------------------------
        super().__init__()
        self.n_frequencies = torch.arange(edge_embed_size//2, dtype=torch.float).to(device)
        self.frequencies = math.pi * div_factor ** self.n_frequencies/min_res
    # =========================================
    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()
# ==============================================
def coord2diff(coordinates, normalize=False):
    coord_diff = coordinates[:, :, None] - coordinates[:, None]
    radial = torch.sum(coord_diff**2, axis=-1).unsqueeze(-1)
    if normalize:
        norm = torch.sqrt(radial).detach() + 1e-8
        coord_diff = coord_diff / norm
    return radial, coord_diff


