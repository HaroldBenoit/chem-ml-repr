
import torch
from typing import Optional, Callable, List
import os
import os.path as osp
from tqdm import tqdm
import pathlib
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data

from utils import download_dataset, data_to_graph

class InMemorySmilesDataset(InMemoryDataset):
    
    def __init__(self, root: str, filename:str, raw_url:str, smiles_column_name:str, target_names: List[str],add_hydrogen: bool, seed: int, transform: Optional[Callable],
                 pre_transform: Optional[Callable],
                 pre_filter: Optional[Callable]):
        
        self.root = root
        self.filename= filename
        self.raw_url= raw_url
        self.smiles_column_name= smiles_column_name
        self.target_names=target_names
        self.add_hydrogen = add_hydrogen
        self.seed = seed
        
        if add_hydrogen:
            p=pathlib.Path(self.root)
            self.root = f"{str(p.parent)}/{p.stem}_hydrogen"
        os.makedirs(f"{self.root}/raw", exist_ok=True)
        os.makedirs(f"{self.root}/processed", exist_ok=True)
        
        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit
            return [self.filename] 
        except ImportError:
            print("rdkit needs to be installed!")
            return []
    

    @property
    def processed_file_names(self) -> str:
        #extracting the file name
        path = pathlib.Path(self.filename)
        stem = path.stem 
        return [f"{stem}.pt"]


    def download(self):
        download_dataset(raw_dir=self.raw_dir, filename=self.filename, raw_url=self.raw_url, target_columns=self.target_names,
                                              smiles_column_name=self.smiles_column_name)
        
    
    def process(self):
        
        """ Full processing procedure for the raw csv dataset. 
        
            1. Read the csv file,excepts a single columns of smiles string, the rest is considered as a target
            For each row:
                2. ETKDG seeded method 3D coordinate generation
                3. QM9 featurization of nodes
                4. Create the complete graph (no self-loops) with covalent bond types as edge attributes
                
            5. Bundle everything into the Data (graph) type
        """
        
        #1. Read the csv file,excepts a single columns of smiles string, the rest is considered as a target
        df = pd.read_csv(self.raw_paths[0],index_col=0, encoding="utf-8")
        target = torch.tensor(df.values)
    
        ## counting the number of failed 3D generations
        failed_counter = 0
        data_list = []
        # iterating over the given range
        data_len = len(df)
        
        aux_data = {"old_data_len":len(df)}
        
        
        
        for idx in tqdm(range(data_len)):
            data_list.append(data_to_graph(data=df.index[idx], y=target[idx].unsqueeze(0), idx=idx,
                                           seed=self.seed, add_hydrogen=self.add_hydrogen, pre_transform=self.pre_transform, pre_filter=self.pre_filter))    
            
        ## removing None
        curr_len = len(data_list)
        data_list= [data for data in data_list if data is not None]
        new_len = len(data_list)
        failed_counter = curr_len - new_len
            
        aux_data["total_num_skipped"]= failed_counter
                   
        print(f"NUM MOLECULES SKIPPED {failed_counter}, {failed_counter/(curr_len):.2f}% of the data")
            
        # saving the auxiliary data
        torch.save(aux_data, osp.join(self.processed_dir, "aux_data.pt"))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
