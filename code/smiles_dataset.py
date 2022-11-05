from typing import Callable, List, Optional, Tuple
import pathlib
import pdb


import rdkit
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



class SmilesDataset(InMemoryDataset):
    
    
    def __init__(self, root: str, filename:str, add_hydrogen=False, seed=0x00ffd, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.add_hydrogen = add_hydrogen
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


    def get_molecule_and_coordinates(self, smile: str) ->  Tuple[Chem.rdchem.Mol, torch.Tensor]:
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
        ps.randomSeed = self.seed
        #ps.coordMap = coordMap = {0:[0,0,0]}
        AllChem.EmbedMolecule(m,ps)


        conf = m.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)

        ## if we dont want hydrogen, we need to rebuild a molecule without explicit hydrogens
        if not(self.add_hydrogen):
            m = Chem.MolFromSmiles(smile)
            
        return m, pos    

        


    def process(self):
        
        df = pd.read_csv(self.raw_paths[0],index_col=0, encoding="utf-8")

        target = torch.tensor(df.values)

        data_list = []
        for idx, smile in enumerate(tqdm(df.index)):

            ## ETKDG seeded method 3D coordinate generation
            mol, pos = self.get_molecule_and_coordinates(smile)


            ## QM9 featurization of nodes

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
            
            
            bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
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
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
            

            #x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
            
            x = torch.tensor([atomic_number, aromatic, sp, sp2, sp3],
                              dtype=torch.float).t().contiguous()
            #x = torch.cat([x1.to(torch.float), x2], dim=-1)

            y = target[idx].unsqueeze(0)

            data = Data(x=x, z=z, pos=pos, edge_index=edge_index,
                        edge_attr=edge_attr, y=y, name=smile, idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            
                   
        torch.save(self.collate(data_list), self.processed_paths[0])


