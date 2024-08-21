A detailed review of recent deep learning advancements, focusing on frameworks that integrate diffusion models and equivariant graph neural networks for protein and small molecule generation, is provided in this [paper][10].

If you use this work please cite my work:

```
@article{soleymani2023protinteract,
  title={ProtInteract: A deep learning framework for predicting protein--protein interactions},
  author={Soleymani, Farzan and Paquet, Eric and Viktor, Herna Lydia and Michalowski, Wojtek and Spinello, Davide},
  journal={Computational and Structural Biotechnology Journal},
  volume={21},
  pages={1324--1348},
  year={2023},
  publisher={Elsevier}
}
```

```
@article{soleymani2024structure,
  title={Structure-based protein and small molecule generation using EGNN and diffusion models: A comprehensive review},
  author={Soleymani, Farzan and Paquet, Eric and Viktor, Herna Lydia and Michalowski, Wojtek},
  journal={Computational and Structural Biotechnology Journal},
  year={2024},
  publisher={Elsevier}
}
```

# PDB Dataset retrieval for training the diffusion model from scratch
This repository contains the codes to retrieve 3-D coordinates of the protein backbone from [PDB database][1]. 
## To prepare the dataset for training PeptiDiff, follow these steps:

1 - Fetch Dataset: The Fetch_Dataset.py script retrieves the necessary dataset from the PDB database and extracts the files into a specified folder.

2 - Process Protein Data: Next, the Data_Processing.py script processes the extracted dataset. It extracts 3-D coordinates of alpha-carbons and assigns physicochemical features to each protein, utilizing functions from AA_features.py. You can customize the padding length of the protein directly within Data_Processing.py. This script saves two dictionaries: one for backbone coordinates and another for backbone features. You can select which features to include by modifying the AA_features.py script.

3 - Utility Functions: The Functions.py script contains essential functions for data processing, normalization, training, and other related tasks.

# Pre-processed datasets for training the models.

The raw datasets, [alpha-carbon coordinates.][2] and the [amino acid sequences of the proteins][3] are available for access.
 
The "AA_features.py" code includes physicochemical features of the 20 standard amino acids. Additionally, the physicochemical features of the unknown amino acid (X) are computed using the median of the known features. Each amino acid can be encoded using their respective [physicochemical features][4] (Table 1). 

Two pre-processed datasets are provided: a [backbone 3-D coordinates dictionary][5] and a [backbone feature dictionary][6] comprising physicochemical properties. For each protein, the data in both dictionaries are normalized and padded to a specified length to ensure uniformity across the dataset.

The [training][7], [validation][8] and [testig][9] datasets are available to download. These datasets are augmented by randomly rotating the original datasets (20 times). 


In order to fetch the dataset from the PDB database, follow these steps:

```
docker build --no-cache -f Dockerfile_data -t dataset_image .
```
```
docker run --rm -v /path/to/host:/app/Code/Dataset dataset_image
```
# Train the diffusion model and generate n-samples of new protein sequences and structures.

If you'd like to use the pre-existing datasets mentioned above, please download the [backbone 3-D coordinates dictionary][5] and [backbone feature dictionary][6], then move them to the 'Dataset' folder within the 'Code' directory. After that, follow these steps to implement the training and inference processes:


```
docker build --no-cache -f Dockerfile_train -t training_inference_image .
```

```
docker run --rm -it --gpus all training_inference_image --Samples 1000 --eta 1
```

This code trains the model and generates 1000 new instances (default - you can increase or decrease it based on the available computational resources) as well as density, coverage and Frechet distance metrics for both generated 3-D coordinates and physicochemical features. This framework implements both DDIM and DDPM which can be selected by changing the parameter (&eta;) where &eta; = 0$ represents DDIM and &eta; = 1$ denotes DDPM.


Similar works using EGNNs typically assign only positional and temporal embeddings as graph node embeddings. In contrast, this work assigns physicochemical features as node embeddings, which undergo a diffusion process, unlike positional and temporal embeddings (in this work, these features are assigned as node attributes - See EGNN code). After completing the training and inference, import the saved features and utilize the Sequence_Builder.py script to construct potential sequences for the generated proteins. The framework creates sequences based on three physicochemical features: hydrophobicity, $\alpha$-helix probability, and solvent-accessible surface area (SASA). These features are assigned as node attributes within the EGNN architecture.


## Physicochemical properties of 20 amino acids

| Amino acid    | Symbol | &alpha;-helix probability | Hydrophobicity | SASA  |
|---------------|--------|----------------------------|----------------|-------|
| Alanine       | A      | 0.42                       | 0.62           | 1.181 |
| Cysteine      | C      | 0.17                       | 0.29           | 1.461 |
| Aspartate     | D      | 0.25                       | -0.90          | 1.587 |
| Glutamate     | E      | 0.42                       | -0.74          | 1.862 |
| Phenylalanine | F      | 0.30                       | 1.19           | 2.228 |
| Glycine       | G      | 0.13                       | 0.48           | 0.881 |
| Histidine     | H      | 0.27                       | -0.40          | 2.025 |
| Isoleucine    | I      | 0.30                       | 1.38           | 1.810 |
| Lysine        | K      | 0.32                       | -1.50          | 2.258 |
| Leucine       | L      | 0.39                       | 1.06           | 1.931 |
| Methionine    | M      | 0.38                       | 0.64           | 2.034 |
| Asparagine    | N      | 0.21                       | -0.78          | 1.655 |
| Proline       | P      | 0.13                       | 0.12           | 1.468 |
| Glutamine     | Q      | 0.36                       | -0.85          | 1.932 |
| Arginine      | R      | 0.36                       | -2.53          | 2.560 |
| Serine        | S      | 0.20                       | -0.18          | 1.298 |
| Threonine     | T      | 0.21                       | -0.05          | 1.525 |
| Valine        | V      | 0.27                       | 1.08           | 1.645 |
| Tryptophan    | W      | 0.32                       | 0.81           | 2.663 |
| Tyrosine      | Y      | 0.25                       | 0.26           | 2.368 |

**Sources:**
- $\alpha$-helix probability: [Holm et al., 1996]([https://doi.org/10.1016/0022-2836(96)00337-7](https://www.science.org/doi/abs/10.1126/science.273.5275.595))
- Hydrophobicity: [Law et al., 2014]([https://doi.org/10.1016/j.jmb.2014.08.017](https://pubs.acs.org/doi/10.1021/jz402762h))
- Solvent-accessible surface area (SASA): [Ausaf et al., 2014]([https://doi.org/10.1016/j.biocel.2014.05.004](https://pubmed.ncbi.nlm.nih.gov/24678666/))



[1]: https://files.wwpdb.org/pub/pdb/data/biounit/PDB/divided/

[2]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=ERw4N-f4U6BNutxBZ67JtbUBF29r45VJifBIzTVFaCvcew&e=79FvMR

[3]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=Eescxh5uKtRGtBtdVZ7BSc8BGGvR9GXhhaw_2mKNKMQtzg&e=EpZjyQ

[4]: https://www.sciencedirect.com/science/article/pii/S2001037023000296

[5]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=EXYcS3XKLYlOqQ8_fYAHa34BoWRzM0MFKZNoEmjOacKZIQ&e=KMqweP

[6]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=EWajMPllOqRDuOFgfvucGREBJApruAi07NXg6VmuXnYX3g&e=ieN7SJ

[7]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=Efrj4s-s7IhHmRmQJahgGUAB3c7AWxCB3LH2sPDcGZqtNw&e=xDl7Hg

[8]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=ERI8YR-laKVEtW75QRXNkCwBtPkJdPAVOjir-eAEXBQPVg&e=1VsVKC

[9]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=ESUT9WMvnnlNjGP3QUqFl1YBKZPS4eEobvftjXk4p0BiRA&e=JbR2j3

[10]: https://www.sciencedirect.com/science/article/pii/S2001037024002228
