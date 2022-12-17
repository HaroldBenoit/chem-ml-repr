import os.path as osp
import torch
from datasets_classes import MatBenchMpIsMetal
from tqdm import tqdm

dataset= MatBenchMpIsMetal(root="../data/matbench/mp_is_metal/")

data = dataset[70575]

print(data)

for data_split_idx in range(50):

    data_list = torch.load(osp.join("../data/matbench/mp_is_metal/processed",f"matbench_mp_is_metal.json_{data_split_idx}.pt"))
    print("split: ", data_split_idx)
    for data in data_list:
        try:
            dist = data.dist
        except:
            print("index: " , data.idx)
            print("data: ", data)
        

        