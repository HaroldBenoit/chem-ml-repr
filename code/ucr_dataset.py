
import torch
from typing import Optional, Callable, List
import os
import os.path as osp
from tqdm import tqdm
import pathlib
import pandas as pd

from torch_geometric.data import Dataset

from utils import download_dataset, data_to_graph


import json
import gzip

from pymatgen.core import Structure

class UcrDataset(Dataset):
    """ Pytorch Geometric dataset for processing of smiles data, better suited for datasets that don't fit into RAM or take up a lot of space.
    """
    
        

    def __init__(self, root: str,filename:str,raw_url:str, data_column_name:str, target_names: List[str],add_hydrogen: bool, seed: int, transform: Optional[Callable],
                 pre_transform: Optional[Callable],
                 pre_filter: Optional[Callable]):
        """
        Args:
            root (str): root directory where the raw data can be found and where the processed data will be stored
            filename (str): csv filename of the dataset
            add_hydrogen (bool, optional): If True, hydrogen atoms will be part of the description of the molecules. Defaults to False.
            seed (hexadecimal, optional): seed for randomness, relevant for 3D coordinates generation. Defaults to 0x00ffd.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
        """
        self.root = root
        self.filename= filename
        self.raw_url= raw_url
        self.data_column_name= data_column_name
        self.target_names=target_names
        self.add_hydrogen = add_hydrogen
        self.seed = seed
        self.split_factor = 50
        
        if add_hydrogen:
            p=pathlib.Path(self.root)
            self.root = f"{str(p.parent)}/{p.stem}_hydrogen"
        os.makedirs(f"{self.root}/raw", exist_ok=True)
        os.makedirs(f"{self.root}/processed", exist_ok=True)
        
        super().__init__(self.root, transform, pre_transform, pre_filter)
        
        
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
        return [f"{stem}_{i}.pt" for i in range(self.split_factor)]


    def download(self):
        download_dataset(raw_dir=self.raw_dir, filename=self.filename, raw_url=self.raw_url, target_columns=self.target_names,
                                              data_column_name=self.data_column_name)
        
    def process(self):
        
        """ Full processing procedure for the raw csv dataset. 
        
            1. Read the csv file,excepts a single columns of smiles string, the rest is considered as a target
            For each row:
                2. ETKDG seeded method 3D coordinate generation
                3. QM9 featurization of nodes
                4. Create the complete graph (no self-loops) with covalent bond types as edge attributes
                
            5. Bundle everything into the Data (graph) type
        """
        
        
        if "csv" in self.filename:
            #1. Read the csv file,excepts a single columns of smiles string, the rest is considered as a target
            # this is the usual situation for smiles dataset
            df = pd.read_csv(self.raw_paths[0],index_col=0, encoding="utf-8")
            original_data = df.index
            target = torch.tensor(df.values)
        elif "json.gz" in self.filename:
            #exppected behaviour when dealing with matbench dataset
            complete_path = f"{self.raw_dir}/{self.filename}"

            with gzip.open(complete_path, 'r') as fin:        #  gzip
                json_bytes = fin.read()                      #  bytes (i.e. UTF-8)

            json_str = json_bytes.decode('utf-8')            # string (i.e. JSON)
            data = json.loads(json_str) 
            
            original_data = [Structure.from_dict(data_list[0]) for data_list in data[self.data_column_name]]
            target = torch.tensor([data_list[1] for data_list in data[self.data_column_name]])
    

        # dashboard: http://127.0.0.1:8787/status
        # setting up the local cluster not to overuse all the cores
        #cpu_count = os.cpu_count()
        #usable_cores = cpu_count//2
        #num_threads_per_worker = max(4, usable_cores//2)
        #n_workers = usable_cores // num_threads_per_worker
        #dask.config.set({'distributed.comm.timeouts.connect': 60, 'distributed.comm.timeouts.tcp': 60, 'distributed.client.heartbeat':10})
        #
        #if self.on_cluster:
        #    cluster = LocalCluster(n_workers=5, threads_per_worker=2, memory_limit=12e9)
        #else:
        #    cluster = LocalCluster(n_workers=3, threads_per_worker=1, memory_limit=1e9)
        #    
        #client = Client(cluster)
        
        ## counting the number of failed 3D generations
        failed_counter = 0
        data_list = []
        # iterating over the given range
        data_len = len(df)
        step = data_len//self.split_factor 

        ## necessary data to log so that we can tell which idx data goes into which split at which point
        aux_data = {"old_data_len":len(df), "step": step, }
        
        for i in tqdm(range(self.split_factor)):
            
            ## making sure we're covering the entire dataset
            if i != self.split_factor -1:
                indexes = range(i*step, min((i+1)*step, data_len))
            else:
                indexes = range(i*step, data_len)
            
            data_list = [data_to_graph(data=original_data[idx], y=target[idx].unsqueeze(0), idx=idx,
                                           seed=self.seed, add_hydrogen=self.add_hydrogen, pre_transform=self.pre_transform, pre_filter=self.pre_filter) for idx in indexes]        
            #data_list = client.compute(allpromises)
            #data_list = client.gather(data_list)
            
            ## need to count the number of skipped molecules to be able to give correct index in get()
            curr_len = len(data_list)
            data_list= [data for data in data_list if data is not None]
            new_len = len(data_list)
            num_skipped= curr_len - new_len
            failed_counter += num_skipped
            
            if i == 0:
                aux_data[0] = {"begin": 0, "end": step - num_skipped}
            elif i > 0 and i < self.split_factor -1:
                last_end = aux_data[i-1]["end"]
                aux_data[i] = {"begin": last_end, "end": last_end + step - num_skipped}
            elif i == self.split_factor-1:
                last_end = aux_data[i-1]["end"]
                # possibly bigger chunk for last dataset
                aux_data[i] = {"begin": last_end, "end": last_end + data_len - i*step - num_skipped}
                
            torch.save(data_list, osp.join(self.processed_dir, self.processed_file_names[i]))
            
        aux_data["total_num_skipped"]= failed_counter
            
                   
        print(f"NUM MOLECULES SKIPPED {failed_counter}, {failed_counter/(data_len):.2f}% of the data")
            
        # saving the auxiliary data
        torch.save(aux_data, osp.join(self.processed_dir, "aux_data.pt"))
    
    def len(self):
        return self.split_factor
    
    def get(self,idx):
        aux_data = torch.load(osp.join(self.processed_dir, "aux_data.pt"))
        old_len = aux_data["old_data_len"]
        total_num_skipped = aux_data["total_num_skipped"]
        
        if idx > (old_len - total_num_skipped) or idx < 0:
            raise IndexError("Index out of range")
        
        data_split_idx = -1
        
        for i in range(self.split_factor):
            if idx >= aux_data[i]["begin"] and idx < aux_data[i]["end"]:
                data_split_idx = i
                break
            
        if data_split_idx == -1:
            raise IndexError(f"Index couldn't be found in one of the {self.split_factor} data splits")
        
        path = pathlib.Path(self.filename)
        stem = path.stem 
        data_list = torch.load(osp.join(self.processed_dir,f"{stem}_{data_split_idx}.pt"))
        
        correct_idx = idx - aux_data[data_split_idx]["begin"]
        return data_list[correct_idx]
                