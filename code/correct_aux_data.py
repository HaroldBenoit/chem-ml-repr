
import torch
import os.path as osp

aux_data = torch.load(osp.join("../data/qm9/processed/", "aux_data.pt"))

step = aux_data["step"]
split_factor=50

for i in range(split_factor):
    curr_end = aux_data[i]["end"]
    if i == 0:
        aux_data[0] = {"begin": 0, "end": curr_end}
    elif i > 0 and i < split_factor -1:
        last_end = aux_data[i-1]["end"]
        aux_data[i] = {"begin": last_end, "end": last_end + curr_end}
    elif i == split_factor-1:
        last_end = aux_data[i-1]["end"]
        # possibly bigger chunk for last dataset
        aux_data[i] = {"begin": last_end, "end": last_end + curr_end}
        
torch.save(aux_data, osp.join("../data/qm9/processed/", "aux_data.pt"))