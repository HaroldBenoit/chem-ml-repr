import pandas as pd
import numpy as np
from smiles_dataset import SmilesInMemoryDataset
from typing import Optional, Callable
import os
import os.path as osp
from torch_geometric.data import (
    download_url,
    extract_zip,
)
from typing import Tuple, List
import pathlib
import argparse


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

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    Dataset)

## parallelization
import os
import dask
dask.config.set(scheduler="processes")
from dask.distributed import Client, LocalCluster

    
    
    #def __init__(self, root: str, add_hydrogen=False, seed=0x00ffd) -> None:
    #    self.root = root
    #    self.add_hydrogen = add_hydrogen
    #    self.seed = seed
    #    self.datasets_func = {"qm9": self.QM9Dataset, "bace":self.BaceDataset, "bbbp":self.BBBPDataset}

    

class QM9Dataset(Dataset):
    """Load QM9 dataset
    QM9 is a comprehensive dataset that provides geometric, energetic,
    electronic and thermodynamic properties for a subset of GDB-17
    database, comprising 134 thousand stable organic molecules with up
    to 9 heavy atoms.  All molecules are modeled using density
    functional theory (B3LYP/6-31G(2df,p) based DFT).
    Random splitting is recommended for this dataset.
    The source data contain:
        - qm9.sdf: molecular structures
        - qm9.sdf.csv: tables for molecular properties
        - "mol_id" - Molecule ID (gdb9 index) mapping to the .sdf file
        - "A" - Rotational constant (unit: GHz)
        - "B" - Rotational constant (unit: GHz)
        - "C" - Rotational constant (unit: GHz)
        - "mu" - Dipole moment (unit: D)
        - "alpha" - Isotropic polarizability (unit: Bohr^3)
        - "homo" - Highest occupied molecular orbital energy (unit: Hartree)
        - "lumo" - Lowest unoccupied molecular orbital energy (unit: Hartree)
        - "gap" - Gap between HOMO and LUMO (unit: Hartree)
        - "r2" - Electronic spatial extent (unit: Bohr^2)
        - "zpve" - Zero point vibrational energy (unit: Hartree)
        - "u0" - Internal energy at 0K (unit: Hartree)
        - "u298" - Internal energy at 298.15K (unit: Hartree)
        - "h298" - Enthalpy at 298.15K (unit: Hartree)
        - "g298" - Free energy at 298.15K (unit: Hartree)
        - "cv" - Heat capavity at 298.15K (unit: cal/(mol*K))
        - "u0_atom" - Atomization energy at 0K (unit: kcal/mol)
        - "u298_atom" - Atomization energy at 298.15K (unit: kcal/mol)
        - "h298_atom" - Atomization enthalpy at 298.15K (unit: kcal/mol)
        - "g298_atom" - Atomization free energy at 298.15K (unit: kcal/mol)
    "u0_atom" ~ "g298_atom" (used in MoleculeNet) are calculated from the
    differences between "u0" ~ "g298" and sum of reference energies of all
    atoms in the molecules, as given in
    https://figshare.com/articles/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643
    """
    
    
    target_names = ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298","h298", "g298"]
    

    def __init__(self, root: str, add_hydrogen=False, seed=0x00ffd, on_cluster:bool = False, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
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
        self.filename="qm9.csv"
        self.raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
        self.add_hydrogen = add_hydrogen
        self.seed = seed
        self.on_cluster = on_cluster
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
        download_dataset(raw_dir=self.raw_dir, filename=self.filename, raw_url=self.raw_url, target_columns=QM9Dataset.target_names,
                                              smiles_column_name="smiles")
        
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
    

        # dashboard: http://127.0.0.1:8787/status
        # setting up the local cluster not to overuse all the cores
        #cpu_count = os.cpu_count()
        #usable_cores = cpu_count//2
        #num_threads_per_worker = max(4, usable_cores//2)
        #n_workers = usable_cores // num_threads_per_worker
        dask.config.set({'distributed.comm.timeouts.connect': 60, 'distributed.comm.timeouts.tcp': 60, 'distributed.client.heartbeat':10})
        
        if self.on_cluster:
            cluster = LocalCluster(n_workers=5, threads_per_worker=2, memory_limit=12e9)
        else:
            cluster = LocalCluster(n_workers=3, threads_per_worker=1, memory_limit=1e9)
            
        client = Client(cluster)
        
        
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
            
            data_list = [smiles_to_graph(smile=df.index[idx], y=target[idx].unsqueeze(0), idx=idx,
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
        
        # first find in which split idx belongs to
        


def BaceDataset(self) -> Tuple[str,str, List[str]]:
    """ 
    Load BACE dataset
    The BACE dataset provides quantitative IC50 and qualitative (binary label)
    binding results for a set of inhibitors of human beta-secretase 1 (BACE-1).
    All data are experimental values reported in scientific literature over the
    past decade, some with detailed crystal structures available. A collection
    of 1522 compounds is provided, along with the regression labels of IC50.
    Scaffold splitting is recommended for this dataset.
    The raw data csv file contains columns below:
    - "mol" - SMILES representation of the molecular structure
    - "pIC50" - Negative log of the IC50 binding affinity
    - "class" - Binary labels for inhibitor  
    """ 
    BACE_REGRESSION_TASKS = "pIC50"
    BACE_CLASSIFICATION_TASKS = "Class"
    filename="bace.csv"
    raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv'
    root, target_names = self.download_dataset(root=self.root, filename=filename, raw_url=raw_url, target_columns=[BACE_CLASSIFICATION_TASKS, BACE_REGRESSION_TASKS],
                                          smiles_column_name="mol", add_hydrogen=self.add_hydrogen)
    return root, filename,target_names
def BBBPDataset(self) -> Tuple[str, str, List[str]]:
    """
    Load BBBP dataset
    The blood-brain barrier penetration (BBBP) dataset is designed for the
    modeling and prediction of barrier permeability. As a membrane separating
    circulating blood and brain extracellular fluid, the blood-brain barrier
    blocks most drugs, hormones and neurotransmitters. Thus penetration of the
    barrier forms a long-standing issue in development of drugs targeting
    central nervous system.
    This dataset includes binary labels for over 2000 compounds on their
    permeability properties.
    Scaffold splitting is recommended for this dataset.
    The raw data csv file contains columns below:
    - "name" - Name of the compound
    - "smiles" - SMILES representation of the molecular structure
    - "p_np" - Binary labels for penetration/non-penetration
    """ 
    BBBP_CLASSIFICATION_TASKS = "p_np"    
    filename="BBBP.csv"
    raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
    root, target_names = self.download_dataset(root=self.root, filename=filename, raw_url=raw_url, target_columns=[BBBP_CLASSIFICATION_TASKS],
                                          smiles_column_name="smiles", add_hydrogen=self.add_hydrogen)
    return root, filename, target_names


def download_dataset(raw_dir:str, filename:str, raw_url:str, target_columns:List[str], smiles_column_name: str):

        complete_path = f"{raw_dir}/{filename}"
        filepath= download_url(raw_url, raw_dir)
        df = pd.read_csv(complete_path)
        col_list=[smiles_column_name]+ target_columns
        df.drop(df.columns.difference(col_list), axis=1, inplace=True)
        df.set_index(smiles_column_name, drop=True, inplace=True)
        df.to_csv(complete_path)

        return 



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
    try:
        m = Chem.AddHs(m)
    except:
        print("can't add hydrogen")
        return None, None
    
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
        # https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/811862b82ce7402b8ba01201a5d0334a%40uni.lu/
        params = Chem.RemoveHsParameters()
        params.removeDegreeZero = True
        m = Chem.RemoveHs(m, params)
        
    return m, pos    


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




def main():
    
    parser = argparse.ArgumentParser(prog="DatasetProcessing", description="Given the dataset name and root path, processes the dataset according to our method")
    parser.add_argument('--dataset', help=f"Name of the dataset to process")
    parser.add_argument('--root', help="path to the root directory where the raw and processed data will be stored")
    parser.add_argument('--hydrogen', action='store_true', help="If flag specified, hydrogens are explicitly described in graph representation.")
    parser.add_argument('--seed', default=0x00ffd, type=int, help="seed for randomness")
    parser.add_argument('--cluster',action='store_true', help="If flag specified, expects to run on lts2gdk0" )
    args = parser.parse_args()

    datasets = QM9Dataset(root=args.root, add_hydrogen=args.hydrogen, seed=args.seed, on_cluster=args.cluster)
    


    #if args.dataset in datasets.datasets_func:
    #    root, filename, target_names = datasets.datasets_func[args.dataset]()
    #    #forcing processing of dataset by calling it
    #    if not(Datasets.BIG_DATASETS[args.dataset]):
    #        _ = SmilesInMemoryDataset(root=root, filename=filename, add_hydrogen=args.hydrogen, seed=args.seed, begin_index=args.begin, end_index=args.end, on_cluster=args.cluster)
    #    else:
    #        _ = 
    #    print(f"Available targets for {args.dataset} are: {target_names}")
    #else:
    #    raise ValueError(f"Given dataset name {args.dataset} is not in the list of available datasets {list(datasets.datasets_func.keys())}")
        

if __name__=="__main__":
    main()
