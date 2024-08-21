import os
import torch
import itertools
from Functions import Functions
from AA_features import features
# ========================================= #
Norm_AAs = Functions(os.path.join(os.getcwd(), 'Dataset')).Normalize_AA()[1]
# ------------------------------------- #
features = ['H_1', 'alpha', 'SASA']
H_samples_ordered = torch.load('H_samples_ordered.pt')
# ------------------------------------- #
def feature_range(Norm_AAs):
    AA_range = {}
    for key, value in Norm_AAs.items():
        if key == 'X':
            continue
        features = []
        for i in range(len(value)):
            features.append([value[i]*0.96, value[i]*1.04])
        AA_range[key] = features
    return AA_range
# ------------------------------------- #
def feature_compare(AA_range, feat_1, feat_2, feat_3):
    Amino_acid = []
    for key, values in AA_range.items():
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
        Amino_Acids.append(feature_compare(feature_range(Norm_AAs), 
                                           H_samples_ordered[i,j,0],
                                           H_samples_ordered[i,j,1],
                                           H_samples_ordered[i,j,2]))
    for i in range(len(Amino_Acids)):
        if isinstance(Amino_Acids[i], list) and len(Amino_Acids[i]) == 0:
            Amino_Acids[i] = ['X']
    Proteins.append(Amino_Acids)
# ------------------------------------- #

# Full_Sequences = list(itertools.product(*Amino_Acids))








