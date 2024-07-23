# PDB Dataset-Protein Backbone Generation
 This repository contains the codes to retrieve 3-D coordinates of the protein backbone from [PDB database][1]. 
 
 You can find the pre-processed dataset containing the [alpha-carbon coordinates.][2] and the [amino acid sequences of the proteins][3]. 
 
The "AA_features.py" code includes physicochemical features of the 20 standard amino acids. Additionally, the physicochemical features of the unknown amino acid (X) are computed using the median of the known features. Each amino acid can be encoded using their respective [physicochemical features][4]. 


[1]: https://files.wwpdb.org/pub/pdb/data/biounit/PDB/divided/

[2]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=ERw4N-f4U6BNutxBZ67JtbUBF29r45VJifBIzTVFaCvcew&e=79FvMR

[3]: https://uottawa-my.sharepoint.com/personal/fsole078_uottawa_ca/_layouts/15/guestaccess.aspx?share=Eescxh5uKtRGtBtdVZ7BSc8BGGvR9GXhhaw_2mKNKMQtzg&e=EpZjyQ

[4]: https://www.sciencedirect.com/science/article/pii/S2001037023000296

## To prepare the dataset for training PeptiDiff, follow these steps:

1 - Fetch Dataset: The Fetch_Dataset.py script retrieves the necessary dataset from the PDB database and extracts the files into a specified folder.

2 - Process Protein Data: Next, the Protein_Backbone_Dataset.py script processes the extracted dataset. It extracts 3-D coordinates of alpha-carbons and assigns physicochemical features to each protein, utilizing functions from AA_features.py. You can customize the padding length of the protein directly within Protein_Backbone_Dataset.py. This script saves two dictionaries: one for backbone coordinates and another for backbone features. You can select which features to include by modifying the AA_features.py script.

3 - Utility Functions: The Functions.py script contains essential functions for data processing, normalization, training, and other related tasks.

To build the Docker image and run the container, which will automatically process the training dataset, follow these steps. Note that the process can be time-consuming and depends on your internet speed.

```
docker build -t peptidiff .
```
```
docker run --rm -v /c/Users/Soleymani/Documents/GitHub/dataset:/app/code/dataset peptidiff
```
