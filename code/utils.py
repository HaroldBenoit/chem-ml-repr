import pandas as pd
import numpy as np
from typing import Optional, Callable
import os
import os.path as osp
from typing import Tuple, List, Dict, Union
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

from pymatgen.core import Structure, Lattice, Molecule
from pymatgen.io.babel import BabelMolAdaptor


import ssl
import sys

from urllib.request import Request, urlopen



def download_url(url: str, folder: str, log: bool = True,
                 filename: Optional[str] = None):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition('/')[2]
        filename = filename if filename[0] == '?' else filename.split('?')[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    os.makedirs(folder,exist_ok=True)

    context = ssl.create_default_context()
    ## precising agent to pass by forbidden access for some datasets (matbench)
    req = Request(url,headers={'User-Agent': 'Mozilla/5.0'})
 
    data = urlopen(req) 
    with open(path, 'wb') as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path




def download_dataset(raw_dir:str, filename:str, raw_url:str, target_columns:List[str], smiles_column_name: str):
    """_summary_

    Args:
        raw_dir (str): _description_
        filename (str): _description_
        raw_url (str): _description_
        target_columns (List[str]): _description_
        smiles_column_name (str): _description_
    """
    complete_path = f"{raw_dir}/{filename}"
    filepath= download_url(raw_url, raw_dir)
    df = pd.read_csv(complete_path)
    col_list=[smiles_column_name]+ target_columns
    df.drop(df.columns.difference(col_list), axis=1, inplace=True)
    df.set_index(smiles_column_name, drop=True, inplace=True)
    df.to_csv(complete_path, encoding="utf-8")
    
    return 



def from_smiles_to_molecule_and_coordinates(smile: str, seed:int, add_hydrogen: bool) ->  Tuple[Chem.rdchem.Mol, torch.Tensor]:
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



def from_structure_to_molecule(struct:Structure, add_hydrogen: bool) -> Chem.rdchem.Mol:

    # we will compute distances directly using the pymatgen structure

    #then the following conversion : pymatgen.Structure -> pymatgen.Molecule -> pybel_mol -> mol file (to retain 3D information) ->  rdkit molecule
    mol = Molecule(species=struct.species, coords=struct.cart_coords)
    adaptor = BabelMolAdaptor(mol).pybel_mol

    #ideally, we would like to give the correct 3D coordinates to the molecule, so we use .mol file
    mol_file = adaptor.write('mol')

    try:
        new_mol = Chem.MolFromMolBlock(mol_file)  
    except:
        print("unable to convert from mol file to rdkit mol")
        return None
    
    if add_hydrogen:
        try:
            new_mol = Chem.AddHs(new_mol)
        except:
            print("unable to add hydrogen")
            return None

    
    return new_mol     



def from_molecule_to_graph(mol:Chem.rdchem.Mol, y:torch.Tensor, pos:torch.Tensor, name:str, idx:int, data: Union[str,Structure]) -> Data:
    
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
    
    if isinstance(data, str):
        data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                edge_attr=edge_attr, y=y, name=name, idx=idx)
    elif isinstance(data, Structure):
        (row, col) = edge_index
        ## getting distances from distance matrix that is aware of mirror images
        dist = torch.tensor(data.distance_matrix[row,col])
        data= Data(x=x, z=z,edge_index=edge_index, edge_attr=edge_attr, y=y, name=name, idx=idx, dist=dist)

    
    return data
    


def data_to_graph(data:Union[str,Structure], y:torch.Tensor, idx: int, seed:int, add_hydrogen:bool, pre_transform: Callable, pre_filter:Callable) -> Data:
    
    
    if isinstance(data,str):
        name= data
        ## 2. ETKDG seeded method 3D coordinate generation
        mol, pos = from_smiles_to_molecule_and_coordinates(smile=data, seed=seed, add_hydrogen=add_hydrogen)
        
    elif isinstance(data,Structure):
        name = data.formula
        ## no need for 3D coordinate generation as crystallographic structure is given
        mol = from_structure_to_molecule(struct=data, add_hydrogen=add_hydrogen)
        pos = None
    
    if mol is None:
        return None
    
    data= from_molecule_to_graph(mol=mol, y=y, pos=pos, name=name, idx=idx)
    
    if pre_filter is not None and not pre_filter(data):
        return None
    
    if pre_transform is not None:
        data = pre_transform(data)
        
    return data    



import torch

from torch_geometric.transforms import BaseTransform


class Distance(BaseTransform):
    r"""Saves the (weighted) Euclidean distance of linked nodes in its edge attributes. 
    (functional name: :obj:`distance`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True')
        weigthed (bool,optional): If set to True, the edge euclidean distance is weighted by the sum of the atoms atomic radius. Defaults to True.
        
        atom_number_to_radius: (Dict[str,float],optional): Dictionary containig the atomic radius (in Angstrom) given an atomic number.
    """
    def __init__(self, norm: bool=True, max_value: float=None, cat: bool=True, weighted: bool=True, atom_number_to_radius: Dict[str,float]=None ):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.weighted=weighted
        self.atom_number_to_radius= atom_number_to_radius
        
        if self.weighted and self.atom_number_to_radius is None:
            raise ValueError("If distance is weighted, the atomic radius dictionary must be provided")

    def __call__(self, data:Data) -> Data:


        pseudo = data.edge_attr
        ## in the case of smiles molecules, we need to compute distances
        ## we need to check for non-presence of dist instead of presence of pos because Data class has always pos attributes
        if not(hasattr(data,'dist')):
            (row, col), pos = data.edge_index, data.pos
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        else:
            ## in the case of pymatgen structures, we have already computed beforehand as we need to be careful with mirror images
            ## this was done using struct.distance_matrix (which gives Euclidean distance)
            dist = data.dist
        
        ## must weigh distance before normalizing as units [Angstrom] need to match
        if self.weighted:
            # z gives us atomic number
            weights=  torch.tensor([(self.atom_number_to_radius[int(data.z[i])]+ self.atom_number_to_radius[int(data.z[j])])/2 for i,j in data.edge_index.T]).view(-1,1)
            #print(weights)
            dist = (dist/weights).view(-1,1)
        

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)
        
#        print(dist.shape, dist)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max}) weighted={self.weighted}')