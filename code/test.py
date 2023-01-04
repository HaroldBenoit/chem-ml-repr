import os.path as osp
import torch
from datasets_classes import MatBenchMpIsMetal, QM9Dataset
from torch_geometric.data import Data
from tqdm import tqdm

#dataset= MatBenchMpIsMetal(root="../data/matbench/mp_is_metal/")
#
#
##dataset = QM9Dataset(root="../data/qm9",add_hydrogen=True)
#
#data = dataset[100]
#
#print(data)
#
#print()
#
#data.x = data.x[:,:10]
#
#print(data)

#dataset= MatBenchMpIsMetal(root="../data/matbench/mp_is_metal/")


import json
import gzip
from pymatgen.core.structure import Structure, Molecule


json_filename = "../data/matbench/mp_gap/raw/matbench_mp_gap.json.gz"


with gzip.open(json_filename, 'r') as fin:        # 4. gzip
    json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
json_file = json.loads(json_str) 

data_gap = json_file["data"]

i = 0

for data_split_idx in range(50):

    dir = "../data/matbench/mp_is_metal/processed"
    file = f"matbench_mp_is_metal.json_{data_split_idx}.pt"
    
    new_dir = "../data/matbench/mp_gap/processed"
    new_file = f"matbench_mp_gap.json_{data_split_idx}.pt"


    data_list = torch.load(osp.join(dir,file))
    print(len(data_list))
    new_data_list = []
    print("split: ", data_split_idx)
    for data in data_list:
        data.y = torch.tensor([data_gap[i][1]]).view(-1,1)
        new_data_list.append(data)
        
        i += 1
        

        
    torch.save(new_data_list, osp.join(new_dir,new_file))
    




#atom_number_to_radius = torch.load("../important_data/atom_number_to_radius.pt")
#
#
#for data_split_idx in range(1):
#
#    dir = "../data/matbench/mp_is_metal/processed"
#    file = f"matbench_mp_is_metal.json_{data_split_idx}.pt"
#    #new_file = f"matbench_mp_is_metal.json_{data_split_idx}_test.pt"
#
#    data_list = torch.load(osp.join(dir,file))
#    new_data_list = []
#    print("split: ", data_split_idx)
#    for data in data_list:
#        dist = data.dist.view(-1,1)
#        z=data.z
#        
#        if dist.numel() > 0:
#            weights=  torch.tensor([(atom_number_to_radius[int(z[i])]+ atom_number_to_radius[int(z[j])])/2 for i,j in data.edge_index.T]).view(-1,1)
#            dist = (dist/weights).view(-1,1)
#            ## normalization
#            dist = dist / torch.max(dist) 
#          
#        ## add distance feature to the rest of the features
#        pseudo = data.edge_attr
#        pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
#        edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
#        
#        data.edge_attr = edge_attr
#        
#        new_data_list.append(data)
#        
#    torch.save(new_data_list, osp.join(dir,file))
    
            
    
    
    
        
            
            
        
        

        