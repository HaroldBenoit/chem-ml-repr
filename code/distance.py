
import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import (
    Data)
from typing import Dict


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
    def __init__(self, norm: bool=True, max_value: float=None, cat: bool=True, weighted: bool=True, atom_number_to_radius: Dict[str,float]=None, dist_present=False):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.weighted=weighted
        self.atom_number_to_radius= atom_number_to_radius
        self.dist_present = dist_present
        
        if self.weighted and self.atom_number_to_radius is None:
            raise ValueError("If distance is weighted, the atomic radius dictionary must be provided")

    def __call__(self, data:Data) -> Data:


            
        if self.dist_present or hasattr(data,'dist'):
            ## in the case of pymatgen structures, we have already computed beforehand as we need to be careful with mirror images
            ## this was done using struct.distance_matrix (which gives Euclidean distance)
            try:
                dist = data.dist.view(-1,1)
            except AttributeError as e:
                print("index: ", data.idx)
                print("data: ", data)
                raise(e)
        else:
            ## in the case of smiles molecules, we need to compute distances
            ## we need to check for non-presence of dist instead of presence of pos because Data class has always pos attributes
            (row, col), pos = data.edge_index, data.pos
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        
        ## must weigh distance before normalizing as units [Angstrom] need to match
        if self.weighted:
            # z gives us atomic number
            weights=  torch.tensor([(self.atom_number_to_radius[int(data.z[i])]+ self.atom_number_to_radius[int(data.z[j])])/2 for i,j in data.edge_index.T]).view(-1,1)
            #print(weights)
            dist = (dist/weights).view(-1,1)
        

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)
        
#        print(dist.shape, dist)
        pseudo = data.edge_attr


        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max}) weighted={self.weighted}')