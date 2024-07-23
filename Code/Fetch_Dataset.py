""" ########## Processing C-alpha files ########## """
import wget
import pickle
import requests
import pdbreader
import numpy as np
import pandas as pd 
from lxml import etree
from io import StringIO
import os, sys, gemmi, json
from Functions import Functions
# ========================================= #
# Define the directory path
directory = os.path.join(os.getcwd(), 'dataset', 'PDB_alpha_C')
# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)  # Use makedirs to create intermediate directories if needed
    print('The new directory is created:', directory)
# ========================================= #
""" ########## Download PDB files ########## """
class Download_PDB:
    def __init__(self, main_url):
        super().__init__()

        self.main_url = main_url
        self.Pop_list = ["/pub/pdb/data/biounit/PDB/divided/", 
                    "https://www.wwpdb.org/ftp/pdb-ftp-sites", "?C=M;O=A","?C=S;O=A","?C=N;O=D"]
        self.Pop_dict = self.Pop_list[1:]
    # ------------------------------------------ #
    def getLinks(self, url):
        print("Getting links from: " + url)
        session = requests.Session()
        page = session.get(url)
        html = page.content.decode("utf-8")
        tree = etree.parse(StringIO(html), parser=etree.HTMLParser())
        refs = tree.xpath("//a")    
        return list(set([link.get('href', '') for link in refs]))
    # ------------------------------------------ #
    def getsubLinks(self,url):
        print("Getting links from: " + url)
        session = requests.Session()
        page = session.get(url)
        html = page.content.decode("utf-8")
        tree = etree.parse(StringIO(html), parser=etree.HTMLParser())
        refs = tree.xpath("//a")    
        return list(set([link.get('href', '') for link in refs]))
    # ------------------------------------------ #
    def Download(self,directory):
        urls = self.getLinks(main_url)
        urls.remove('/pub/pdb/data/biounit/PDB/')
        remove_links = []
        for i in self.Pop_dict:
            remove_links.append(main_url+i)
        Links = {}
        for url in urls:
            Suburl = []
            for suburl in self.getsubLinks(main_url+url):
                Suburl.append(suburl)
            for item in self.Pop_list:
                if item in Suburl:
                    Suburl.remove(item)
            Links[url] = Suburl
            if url in self.Pop_dict:
                del Links[url]
        # ------------------------------------------
        for url, codes in Links.items():
            for code in codes:
                if main_url+url+code in remove_links:
                    continue
                else:
                    wget.download(main_url+url+code, directory)
    # ------------------------------------------



""" ########## Extract_Coordinates ########## """
def Extract_Coordinates(directory):
    print('*****************************************')
    print('Extracting alpha-Carbon 3-D coordinates!')
    print('*****************************************')
    extensions = [".pdb"+str(i) for i in range(1,32)]
    # extension = 'pdb1'
    os.chdir(directory)
    chain = {}
    protein = {}
    AA_Chain = {}
    Ca_Chain = {}
    for item in tqdm(os.listdir(directory)):
        for extension in extensions:
            if item.endswith(extension):
                # Parse and get basic information
                id_ = item.replace(extension, "")
                protein[id_] = pdbreader.read_pdb(item)
                if 'ATOM' not in protein[id_].keys():
                    continue
                else:
                    chain[id_] = set(list(protein[id_]['ATOM']['chain']))
                    ca = {}
                    c_a = {}
                    seq = {}
                    for ch in chain[id_]:
                        ca[ch] = protein[id_]['ATOM'].loc[
                            protein[id_]['ATOM']['name'] == 'CA',
                            ['chain','x','y','z','resname']].loc[
                            protein[id_]['ATOM'].loc[protein[id_]['ATOM']['name'] == 'CA',
                            ['chain','x','y','z','resname']]['chain']==ch].drop('chain', axis=1)
                        # Filter sequence with length bigger than 5 and less than 1024
                        if ca[ch].shape[0] <= 3:
                            continue
                        else:
                            seq[ch] = list(ca[ch]['resname'])
                            c_a[ch] = ca[ch].drop('resname', axis=1).to_numpy()
                    # ---------------------------------------------------------- #   
                        ID = id_+'_'+ch
                        Ca_Chain[ID] = c_a[ch]
                        # AA_Chain[ID] = seq[ch]
    # -------------------------------------------------------------------------- #   
    Info = dict()
    # Info['AA_sq'] = AA_Chain
    Info['Coordinate'] = Ca_Chain
    return Info

""" ########## Save the Dataset ########## """

if __name__ == "__main__":
    main_url = 'https://files.wwpdb.org/pub/pdb/data/biounit/PDB/divided/'
    directory = os.path.join(os.getcwd(), 'dataset', 'PDB_alpha_C')
    print('======================================')
    print('Retrieving proteins from PDB database!')
    print('======================================')
    LINKS = Download_PDB(main_url).Download(directory)
    Extract_dataset = Functions(directory).gz_extract()
    Proteins = Extract_Coordinates(directory)
    with open("PDB_Backbone_Coordinates.pkl","wb", encoding='utf-8') as file:
        print("saving alpha-Carbon 3-D coordinates as dictionary")
        pickle.dump(Proteins,file)
        file.close()

