import os.path as osp
import torch
from datasets_classes import MatBenchMpIsMetal
from torch_geometric.data import Data
from tqdm import tqdm

#dataset= MatBenchMpIsMetal(root="../data/matbench/mp_is_metal/")

atom_number_to_radius = torch.load("../important_data/atom_number_to_radius.pt")


for data_split_idx in range(50):

    dir = "../data/matbench/mp_is_metal/processed"
    file = f"matbench_mp_is_metal.json_{data_split_idx}.pt"
    #new_file = f"matbench_mp_is_metal.json_{data_split_idx}_test.pt"

    data_list = torch.load(osp.join(dir,file))
    new_data_list = []
    print("split: ", data_split_idx)
    for data in data_list:
        dist = data.dist.view(-1,1)
        z=data.z
        
        if dist.numel() > 0:
            weights=  torch.tensor([(atom_number_to_radius[int(z[i])]+ atom_number_to_radius[int(z[j])])/2 for i,j in data.edge_index.T]).view(-1,1)
            dist = (dist/weights).view(-1,1)
            ## normalization
            dist = dist / torch.max(dist) 
          
        ## add distance feature to the rest of the features
        pseudo = data.edge_attr
        pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
        edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        
        data.edge_attr = edge_attr
        
        new_data_list.append(data)
        
    torch.save(new_data_list, osp.join(dir,file))
    
            
    
    
    
        
            
            
        
        

        