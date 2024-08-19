import os
import torch
from Functions import Functions
from AA_features import features
# ========================================= #
Norm_props, Norm_AAs = Functions(os.path.join(os.getcwd(), 'Dataset')).Normalize_AA()
# ------------------------------------- #
features = ['H_1', 'alpha', 'SASA']
H_samples_ordered = torch.load('H_samples_ordered.pt')
# ------------------------------------- #
# +/- 10% margine for each feature
AA_range = {}
for key, value in Norm_AAs.items():
    AA_range[key] = [[value[0]*0.97, value[0]*1.03], 
                     [value[1]*0.97, value[1]*1.03],
                     [value[2]*0.97, value[2]*1.03]]
# ------------------------------------- #
def feature_compare(AA_range, feat_1, feat_2, feat_3):
    Amino_acid = []
    for key, values in AA_range.items():
        # if feat_2 >= values[1][0] and feat_2 <= values[1][1]:
        if feat_1 >= values[0][0] and feat_1 <= values[0][1] or\
           feat_2 >= values[1][0] and feat_2 <= values[1][1] or\
           feat_3 >= values[2][0] and feat_3 <= values[2][1]:
            Amino_acid.append(key)
    return Amino_acid
# ------------------------------------- #
Proteins = []
for i in range(H_samples_ordered.shape[0]):
    Amino_Acids = []
    for j in range(H_samples_ordered.shape[1]):
        Amino_Acids.append(feature_compare(AA_range, 
                                           H_samples_ordered[i,j,0],
                                           H_samples_ordered[i,j,1],
                                           H_samples_ordered[i,j,2]))
    Proteins.append(Amino_Acids)
# ------------------------------------- #



