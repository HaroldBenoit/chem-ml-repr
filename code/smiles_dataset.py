from typing import Callable, List, Optional, Tuple
import pathlib

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType

import torch
import torch.nn.functional as F
from tqdm import tqdm

from collections import defaultdict
import pandas as pd

from torch_geometric.data import (
    Data,
    InMemoryDataset)

## parallelization
import os
import dask
dask.config.set(scheduler="processes")
from dask.distributed import Client, LocalCluster

class SmilesDataset(InMemoryDataset):
    """ Dataset class to go from (smiles,target) data format to (graph data, target) data format,
    Implemented as a Pytorch Geometric InMemoryDataset for ease of use.
    Hyper-parallelized using Dask, beware."""
    
    
    def __init__(self, root: str, filename:str, add_hydrogen=False, seed=0x00ffd, begin_index:int=0, end_index:int = -1, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        """
q
        Args:
            root (str): root directory where the raw data can be found and where the processed data will be stored
            filename (str): csv filename of the dataset
            add_hydrogen (bool, optional): If True, hydrogen atoms will be part of the description of the molecules. Defaults to False.
            seed (hexadecimal, optional): seed for randomness, relevant for 3D coordinates generation. Defaults to 0x00ffd.
            begin_index (int, optional): beginning index of the processing in the raw data. Defaults to 0 (beginning of the data)
            end_index (int, optional): end index of the processing in the raw data. Defaults to -1 (end of the data)
            
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
        """
        self.add_hydrogen = add_hydrogen
        self.begin_index = begin_index
        self.end_index = end_index
        self.seed = seed
        self.raw_file_names = filename
        if add_hydrogen:
            root = f"{root}_hydrogen"
        
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit
            return [self.__raw_file_names] 
        except ImportError:
            print("rdkit needs to be installed!")
            return []
        
    @raw_file_names.setter
    def raw_file_names(self,value) -> None:
        self.__raw_file_names= value

    @property
    def processed_file_names(self) -> str:
        #extracting the file name
        path = pathlib.Path(self.__raw_file_names) 
        return f"{path.stem}.pt"

    def download(self):
        pass

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
        
        if self.begin_index < 0 or self.begin_index >= len(df):
            raise ValueError(f"begin index value: {self.begin_index} is out of bounds [0, {len(df) -1}]")

        if abs(self.end_index) >= len(df):
            raise ValueError(f"end index value: {self.end_index} is out of bounds [-{len(df) -1}], {len(df) -1}]")
            
        ## translate back from negative indexing to postive indexing
        if self.end_index < 0:
            self.end_index= len(df) + self.end_index + 1
            

        # dashboard: http://127.0.0.1:8787/status
        # setting up the local cluster not to overuse all the cores
        cpu_count = os.cpu_count()
        usable_cores = cpu_count //2
        cluster = LocalCluster(n_workers=usable_cores//2, threads_per_worker=usable_cores//2)
        client = Client(cluster)
        
        
        ## counting the number of failed 3D generations
        failed_counter = 0
        data_list = []
        # iterating over the given range
        data_len = len(df.index[self.begin_index: self.end_index])
        indexes = list(range(self.begin_index, self.end_index))
        
        
        split_factor = 10
        step = data_len//10
        final_data_list=[]
        
        for i in tqdm(range(self.begin_index, self.end_index, step)):
            
            ## making sure we're not out-of-bounds at the beginning
            indexes = range(i, min(i+step, self.end_index))
            allpromises = [smiles_to_graph(smile=df.index[idx], y=target[idx].unsqueeze(0), idx=idx,
                                           seed=self.seed, add_hydrogen=self.add_hydrogen, pre_transform=self.pre_transform, pre_filter=self.pre_filter) for idx in indexes]        
            data_list = dask.compute(allpromises)[0]
            curr_len = len(data_list)
            data_list= [data for data in data_list if data is not None]
            new_len = len(data_list)
            failed_counter += curr_len - new_len
            final_data_list = final_data_list + data_list
            
            
    
        #for idx in tqdm(indexes):
        #    
        #    smile = df.index[idx]
        #    y = target[idx].unsqueeze(0)
        #    data = self.smiles_to_graph(smile=smile, y=y,idx=idx)
        #    
        #    if smile is None:
        #        failed_counter+=1
        #    else:
        #        data_list.append(data)
            
            
        print(f"NUM MOLECULES SKIPPED {failed_counter}, {failed_counter/(data_len):.2f}% of the data")
            
                   
        torch.save(self.collate(final_data_list), self.processed_paths[0])
        
        

def get_molecule_and_coordinates(smile: str, seed:int, add_hydrogen: bool) ->  Tuple[Chem.rdchem.Mol, torch.Tensor]:
    """From smiles string, generate 3D coordinates using seeded ETKDG procedure.
    Args:
        smile (str): valid smiles tring of molecule
    Returns:
        Tuple[Chem.rdchem.Mol, torch.Tensor]: molecule and 3D coordinates of atom
    """
    
    try:
        m = Chem.MolFromSmiles(smile)
    except:
        print("invalid smiles string")
        return None, None
    
    # necessary to add hydrogen for consistent conformer generation
    m = Chem.AddHs(m)
    ## 3D conformer generation
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = seed
    #ps.coordMap = coordMap = {0:[0,0,0]}
    err = AllChem.EmbedMolecule(m,ps)
    
    # conformer generation failed for some reason (molecule too big is an option)
    if err !=0:
        return None, None
    conf = m.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)
    ## if we dont want hydrogen, we need to rebuild a molecule without explicit hydrogens
    if not(add_hydrogen):
        m = Chem.RemoveHs(m)
        
    return m, pos    


@dask.delayed
def smiles_to_graph(smile:str, y:torch.tensor, idx: int, seed:int, add_hydrogen:bool, pre_transform: Callable, pre_filter:Callable) -> Data:
    
    ## 2. ETKDG seeded method 3D coordinate generation
    mol, pos = get_molecule_and_coordinates(smile=smile, seed=seed, add_hydrogen=add_hydrogen)
    
    if mol is None:
        return None
    # 3. QM9 featurization of nodes
    N = mol.GetNumAtoms()
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
    z = torch.tensor(atomic_number, dtype=torch.long)
    
    
    
    # 4. Create the complete graph (no self-loops) with covalent bond types as edge attributes
    
    # must start at 1, as we will be using a defaultdict with default value of 0 (indicating no covalent bond)
    bonds = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
    # getting all covalent bond types
    bonds_dict = {(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()):bonds[bond.GetBondType()] for bond in mol.GetBonds()}
    # returns 0 for all pairs of atoms with no covalent bond 
    bonds_dict = defaultdict(int, bonds_dict)
    # making the complete graph
    first_node_index = []
    second_node_index = []
    edge_type=[]
    distances=[]
    
    for i in range(N):
        for j in range(N):
            if i!=j:
                first_node_index.append(i)
                second_node_index.append(j)
                edge_type.append(bonds_dict[(i,j)] if i < j else bonds_dict[(j,i)])

    edge_index = torch.tensor([first_node_index, second_node_index], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)
    
    #x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    
    # 5. Bundling everything into the Data (graph) type
    
    x = torch.tensor([atomic_number, aromatic, sp, sp2, sp3],dtype=torch.float).t().contiguous()
    #x = torch.cat([x1.to(torch.float), x2], dim=-1)
    data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, y=y, name=smile, idx=idx)
    if pre_filter is not None and not pre_filter(data):
        return None
    
    if pre_transform is not None:
        data = pre_transform(data)
        
    return data    