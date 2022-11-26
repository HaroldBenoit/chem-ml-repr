from smiles_dataset import SmilesInMemoryDataset
from smiles_lightning_data_module import SmilesDataModule
from lightning_model import LightningClassicGNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.transforms import distance
from torch_geometric.loader import DataLoader
import os
import torch
# making sure we are as determinstic as possibe
#torch.use_deterministic_algorithms(True)
import numpy as np

from torch_geometric.data import Data
from typing import List, Callable
from functools import partial
from smiles_dataset import SmilesInMemoryDataset
from torch_geometric.transforms import Compose, distance
from datasets_classes import QM9Dataset
import wandb
import pdb
import argparse

def main():
    
    parser = argparse.ArgumentParser(prog="Training", description="Training pipeline")
    parser.add_argument('--debug', action='store_true', help="If flag specified, activate breakpoints in the script")
    args = parser.parse_args()

    debug= args.debug

    ## dataset
    root= "../data/qm9"
    filename="qm9.csv"
    target='u0'
    hydrogen=False
    classification=False
    output_dim = 2 if classification else 1
    seed=42
    
    ## model
    num_hidden_features=32
    dropout_p = 0.0
    ## pytorch lighting takes of seeding everything
    pl.seed_everything(seed=seed, workers=True)
    
    ## training
    project="test-project"
    run_name="test-qm9"
    num_epochs=1
    # (int) log things every N batches
    log_freq=3
    accelerator="gpu"
    devices = [0]
    
    
    if debug:
        pdb.set_trace(header="Before dataset transform")
    
    # filtering out irrelevant target and computing euclidean distances between each vertices
    
    if "qm9" in filename:
        transforms=Compose([filter_target(target_names=QM9Dataset.target_names, target=target), distance.Distance()])
        dataset = QM9Dataset(root=root, add_hydrogen=hydrogen, seed=seed,transform=transforms)
        

    if debug:
        pdb.set_trace(header="After dataset transform")
    
    # from torch dataset, create lightning data module to make sure training splits are always done the same ways
    data_module = SmilesDataModule(dataset=dataset, seed=seed)
    
    num_node_features = data_module.num_node_features
    num_edge_features= data_module.num_edge_features
    
    gnn_model = LightningClassicGNN(classification=classification, output_dim=output_dim, dropout_p=dropout_p,num_hidden_features=num_hidden_features,  num_node_features=num_node_features, num_edge_features=num_edge_features)
    
    
    #docs: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html#pytorch_lightning.loggers.WandbLogger
    wandb_logger = WandbLogger(save_dir="../training_artifacts/", log_model=True, project=project, name=run_name)
    ## log histograms of gradients and parameters
    # wandb_logger.watch(gnn_model, log_freq=log_freq)
    trainer = pl.Trainer(logger=wandb_logger, deterministic=False, default_root_dir="../training_artifacts/", precision=16,
	 max_epochs=num_epochs, log_every_n_steps=log_freq, devices=devices, accelerator=accelerator)

    # strategy="ddp"   
    
    if debug:
        pdb.set_trace(header="After trainer instantation")
    
    # tune to find the learning rate
    #trainer.tune(gnn_model,datamodule=data_module)
    
    # we can resume from a checkpoint using trainer.fit(ckpth_path="some/path/to/my_checkpoint.ckpt")
    trainer.fit(gnn_model, datamodule=data_module)
    
    if debug:
        pdb.set_trace(header="After trainer fit")
    
    trainer.validate(gnn_model, datamodule=data_module)
    
    wandb.finish()
    
    
    
    
def filter_target(target_names:List[str], target:str)-> Callable[[Data],Data]:
    """ Transform to be given to SmilesInMemoryDataset, has the effect of filtering out all irelevant targets in the Data objects in the dataset at runtime
    Example: for BACE, target_names=['Class', 'PIC50'], we want to train a classifier => target='Class'
    """
    target_idx = target_names.index(target)

    return partial(filter_target_with_idx, target_idx=target_idx)

def filter_target_with_idx(graph:Data, target_idx:int) -> Data:
    new_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, pos=graph.pos ,y=graph.y[:,target_idx:target_idx+1], z=graph.z, name=graph.name, idx=graph.idx) 

    return new_graph



if __name__=="__main__":
    main()
