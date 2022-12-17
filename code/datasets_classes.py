import pandas as pd
import numpy as np
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
from in_mem_ucr_dataset import InMemoryUcrDataset
from ucr_dataset import UcrDataset


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
#import dask
#dask.config.set(scheduler="processes")
#from dask.distributed import Client, LocalCluster


from utils import download_dataset, data_to_graph
    
    
    #def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED) -> None:
    #    self.root = root
    #    self.add_hydrogen = add_hydrogen
    #    self.seed = seed
    #    self.datasets_func = {"qm9": self.QM9Dataset, "bace":self.BaceDataset, "bbbp":self.BBBPDataset}

    

GLOBAL_SEED=0x00ffd



class QM9Dataset(UcrDataset):
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
    is_classification={target:False for target in target_names}

    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        

        filename="qm9.csv"
        raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
        data_column_name="smiles"

        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name, target_names= QM9Dataset.target_names,
                     add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)



class MUVDataset(UcrDataset):
    """Load MUV dataset
    The Maximum Unbiased Validation (MUV) group is a benchmark dataset selected
    from PubChem BioAssay by applying a refined nearest neighbor analysis.
    The MUV dataset contains 17 challenging tasks for around 90 thousand
    compounds and is specifically designed for validation of virtual screening
    techniques.
    Scaffold splitting is recommended for this dataset.
    The raw data csv file contains columns below:
    - "mol_id" - PubChem CID of the compound
    - "smiles" - SMILES representation of the molecular structure
    - "MUV-XXX" - Measured results (Active/Inactive) for bioassays
    """
    
    target_names = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689', 'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810', 'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    is_classification={target:True for target in target_names}
    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        
        filename="muv.csv.gz"
        raw_url= "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"
        data_column_name="smiles"

        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name, target_names= MUVDataset.target_names,
                     add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
        
        
class MatBenchMpIsMetal(UcrDataset):
    """Load matchbench_mp_is_metal dataset
        The raw data contains a compressed json file containing informations on structure and target
    """
    
    target_names = ['is_metal']
    is_classification={target:True for target in target_names}
    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        
        filename="matbench_mp_is_metal.json.gz"
        raw_url= "https://ml.materialsproject.org/projects/matbench_mp_is_metal.json.gz"
        data_column_name="data"

        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name, target_names= MatBenchMpIsMetal.target_names,
                     add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
        
class MatBenchMpGap(UcrDataset):
    """Load matchbench_mp_gap dataset
        The raw data contains a compressed json file containing informations on structure and target
    """
    
    target_names = ['gap']
    is_classification={target:False for target in target_names}
    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        
        filename="matbench_mp_gap.json.gz"
        raw_url= "https://ml.materialsproject.org/projects/matbench_mp_gap.json.gz"
        data_column_name="data"

        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name, target_names= MatBenchMpGap.target_names,
                     add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
        

        
        
class MatBenchMpEForm(UcrDataset):
    """Load matchbench_mp_e_form dataset
        The raw data contains a compressed json file containing informations on structure and target
    """
    
    target_names = ['e_form']
    is_classification={target:False for target in target_names}
    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        
        filename="matbench_mp_e_form.json.gz"
        raw_url= "https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz"
        data_column_name="data"

        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name, target_names= MatBenchMpEForm.target_names,
                     add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        

class BaceDataset(InMemoryUcrDataset):
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
    
    target_names = ['Class','pIC50']
    is_classification={'Class':True, 'pIC50':False}

    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        
        filename="bace.csv"
        raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv'
        data_column_name="mol"
        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name,
                         target_names= BaceDataset.target_names,add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
        
class FreeSolvDataset(InMemoryUcrDataset):
    """Load Freesolv dataset
    The FreeSolv dataset is a collection of experimental and calculated hydration
    free energies for small molecules in water, along with their experiemental values.
    Here, we are using a modified version of the dataset with the molecule smile string
    and the corresponding experimental hydration free energies.
    Random splitting is recommended for this dataset.
    The raw data csv file contains columns below:
    - "smiles" - SMILES representation of the molecular structure
    - "y" - Experimental hydration free energy
    """

    target_names = ['y']
    is_classification={'y':False}

    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
                
        filename="freesolv.csv.gz"
        raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/freesolv.csv.gz'
        data_column_name="smiles"
        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name,
                         target_names= FreeSolvDataset.target_names,add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        

class BBBPDataset(InMemoryUcrDataset):
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

    target_names = ['p_np']
    is_classification={'p_np':True}
    
    def __init__(self, root: str, add_hydrogen=False, seed=GLOBAL_SEED, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
                
        filename="BBBP.csv"
        raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
        data_column_name="smiles"
        super().__init__(root=root, filename = filename, raw_url=raw_url, data_column_name=data_column_name,
                         target_names= BBBPDataset.target_names,add_hydrogen=add_hydrogen, seed=seed, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)



dataset_dict= {"qm9": QM9Dataset, "bace":BaceDataset, "bbbp": BBBPDataset, "freesolv":FreeSolvDataset, "muv":MUVDataset, "mp_is_metal":MatBenchMpIsMetal, "mp_gap":MatBenchMpGap,"mp_e_form":MatBenchMpEForm}


def main():
        
    parser = argparse.ArgumentParser(prog="DatasetProcessing", description="Given the dataset name and root path, processes the dataset according to our method")
    parser.add_argument('--dataset', help=f"Name of the dataset to process. List of available datasets {list(dataset_dict.keys())}")
    parser.add_argument('--root', help="path to the root directory where the raw and processed data will be stored")
    parser.add_argument('--hydrogen', action='store_true', help="If flag specified, hydrogens are explicitly described in graph representation.")
    parser.add_argument('--seed', default=GLOBAL_SEED, type=int, help="seed for randomness")
    args = parser.parse_args()

    


    if args.dataset in dataset_dict:
        dataset_class= dataset_dict[args.dataset]
        #forcing processing of dataset by calling it
        dataset = dataset_class(root=args.root, add_hydrogen=args.hydrogen, seed=args.seed)
        print(f"Available targets for {args.dataset} are: {dataset_class.target_names}")
    else:
        raise ValueError(f"Given dataset name {args.dataset} is not in the list of available datasets {list(dataset_dict.keys())}")
        

if __name__=="__main__":
    main()




