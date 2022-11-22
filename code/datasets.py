import pandas as pd
import numpy as np
from smiles_dataset import SmilesDataset
from typing import Optional, Callable
import os
from torch_geometric.data import (
    download_url,
    extract_zip,
)

import argparse

DATASET_NAMES=['qm9','bace']


def main():
    parser = argparse.ArgumentParser(prog="DatasetProcessing", description="Given the dataset name and root path, processes the dataset according to our method")
    parser.add_argument('--dataset', help=f"name of the dataset to process. Currently available datasets are:\n{DATASET_NAMES}")
    parser.add_argument('--root', help="path to the root directory where the raw and processed data will be stored")
    parser.add_argument('--hydrogen', action='store_true', help="If flag specified, hydrogens are explicitly described in graph representation.")
    parser.add_argument('--seed', default=0x00ffd, type=int, help="seed for randomness")
    parser.add_argument('--begin', default=0, type=int, help="beginning index in the raw data to specify the starting point of processing")
    parser.add_argument('--end', default=-1, type=int, help="beginning index in the raw data to specify the starting point of processing")
    
    args = parser.parse_args()
    print(args)
    parser.add_argument("--regression", action='store_true')
    args = parser.parse_args()
    print(args)
    
    #QM9Dataset(root=args.root, add_hydrogen=args.hydrogen, seed=args.seed, begin_index=args.begin, end_index=args.end)
    



def QM9Dataset(root: str, add_hydrogen=False, seed=0x00ffd, begin_index:int=0, end_index:int = -1,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None) -> SmilesDataset:
    
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
    QM9_TASKS = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
        "h298", "g298"]  
                    
    filename="qm9.csv"
    raw_url= 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
    raw_dir = f"{root}/raw"
    os.makedirs(raw_dir, exist_ok=True)
    complete_path = f"{raw_dir}/{filename}"
    
    ## if data has not been downloaded yet
    if not(os.path.exists(complete_path)):

        filepath= download_url(raw_url, raw_dir)
        df = pd.read_csv(complete_path)
        df.drop(columns=df.columns.difference(QM9_TASKS + ["smiles"]), inplace=True)
        df.set_index("smiles", drop=True, inplace=True)
        df.to_csv(complete_path)
    else:
        df = pd.read_csv(complete_path,index_col=0)
        
    dataset = SmilesDataset(root=root, filename=filename, add_hydrogen=add_hydrogen, seed=seed, begin_index=begin_index, end_index=end_index,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
    target_names= list(df.columns)
    
    return dataset, target_names


def BaceDataset(root: str, add_hydrogen=False, seed=0x00ffd, begin_index:int=0, end_index:int = -1, regression=True,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                pre_filter: Optional[Callable] = None) -> SmilesDataset:
    
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
    raw_dir = f"{root}/raw"
    os.makedirs(raw_dir, exist_ok=True)
    complete_path = f"{raw_dir}/{filename}"
    
    ## if data has not been downloaded yet
    if not(os.path.exists(complete_path)):

        filepath= download_url(raw_url, raw_dir)
        df = pd.read_csv(complete_path)
        col_list=["mol", BACE_CLASSIFICATION_TASKS, BACE_REGRESSION_TASKS]
        df.drop(df.columns.difference(col_list), axis=1, inplace=True)
        df.set_index("mol", drop=True, inplace=True)
        df.to_csv(complete_path)
    else:
        df = pd.read_csv(complete_path,index_col=0)       
        
    dataset = SmilesDataset(root=root, filename=filename, add_hydrogen=add_hydrogen, seed=seed, begin_index=begin_index, end_index=end_index,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
    
    target_names= list(df.columns)
    
    return dataset, target_names 



if __name__=="__main__":
    main()