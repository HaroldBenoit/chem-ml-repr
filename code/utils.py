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
    Data)






def download_dataset(raw_dir:str, filename:str, raw_url:str, target_columns:List[str], smiles_column_name: str):

        complete_path = f"{raw_dir}/{filename}"
        filepath= download_url(raw_url, raw_dir)
        df = pd.read_csv(complete_path)
        col_list=[smiles_column_name]+ target_columns
        df.drop(df.columns.difference(col_list), axis=1, inplace=True)
        df.set_index(smiles_column_name, drop=True, inplace=True)
        df.to_csv(complete_path, encoding="utf-8")

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
