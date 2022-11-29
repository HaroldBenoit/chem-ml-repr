from smiles_lightning_data_module import SmilesDataModule
from lightning_model import LightningClassicGNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
from torch_geometric.transforms import Compose
from utils import Distance
from datasets_classes import QM9Dataset
import wandb
import pdb
import argparse
import datasets_classes

def main():
    
    parser = argparse.ArgumentParser(prog="Training", description="Training pipeline")
    parser.add_argument('--debug', action='store_true', help="If flag specified, activate breakpoints in the script")
    parser.add_argument('--cluster', action='store_true' ,help="If flag specified, we are training on the cluster")
    parser.add_argument('--hydrogen', action='store_true' ,help="If flag specified, we are using the hydrogen dataset")
    parser.add_argument('--checkpoint', help="Path to the checkpoint of the model (ends with .ckpt). Defaults to None")
    
    ##dataset
    parser.add_argument('--root', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--weighted', action="store_true", help="If flag specified, make the edge distances weighted by atomic radius")
    
    ##training
    parser.add_argument('--epochs', default=100)
    args = parser.parse_args()

    debug= args.debug

    ## dataset
    root= args.root
    dataset=args.dataset
    target=args.target

    seed=42
    

    ## pytorch lighting takes of seeding everything
    pl.seed_everything(seed=seed, workers=True)
    
    ## training

    num_epochs=args.epochs
    # (int) log things every N batches
    log_freq=1
    accelerator="gpu"
	#good devices are 0,1,2,3 on the cluster
    if args.cluster:
        devices = [0,1,3]
    else:
        devices = [0]
    
    
    if debug:
        pdb.set_trace(header="Before dataset transform")
    

    
    
    ## DATASET
    dataset_class = datasets_classes.dataset_dict[dataset]
    # filtering out irrelevant target and computing euclidean distances between each vertices
    weighted = args.weighted
    atom_number_to_radius = None if not(weighted) else torch.load("../important_data/atom_number_to_radius.pt")
    distance = Distance(weighted=weighted,atom_number_to_radius=atom_number_to_radius)
    transforms=Compose([filter_target(target_names=dataset_class.target_names, target=target), distance])
    dataset = dataset_class(root=root, add_hydrogen=args.hydrogen,transform=transforms)
    

    if debug:
        pdb.set_trace(header="After dataset transform")
    
    # from torch dataset, create lightning data module to make sure training splits are always done the same ways
    data_module = SmilesDataModule(dataset=dataset, seed=seed)
    
    ## MODEL
    num_hidden_features=256
    dropout_p = 0.0
    classification= dataset_class.is_classification[target]
    output_dim = 2 if classification else 1
    
    num_node_features = data_module.num_node_features
    num_edge_features= data_module.num_edge_features
    
    gnn_model = LightningClassicGNN(seed=seed, classification=classification, output_dim=output_dim, dropout_p=dropout_p,
                                    num_hidden_features=num_hidden_features,  num_node_features=num_node_features, num_edge_features=num_edge_features)
    
    
    ## WANDB
      
    project=f"{dataset}-project"
    run_name=f"target_{target}" 
    #quirk of wandbs
    if "mu" in target:
        run_name=f"target__{target}"
        
    if args.hydrogen:
        run_name=f"{run_name}_hydrogen"
        
    if args.weighted:
        run_name=f"{run_name}_weighted"
        
    
    #docs: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html#pytorch_lightning.loggers.WandbLogger
    wandb_logger = WandbLogger(save_dir="../training_artifacts/", log_model=True, project=project, name=run_name, id=run_name)
    ## log histograms of gradients and parameters
    wandb_logger.watch(gnn_model, log_freq=log_freq)
    
    ## setting up the trainer
    strategy = "ddp" if args.cluster else None
    ## creating early stop callback to ensure we don't overfit
    early_stop_callback = EarlyStopping(monitor="loss/valid", mode="min", patience=num_epochs//2, min_delta=0.00)
    
    trainer = pl.Trainer(logger=wandb_logger, deterministic=False, default_root_dir="../training_artifacts/", precision=16,
	 strategy=strategy,max_epochs=num_epochs ,log_every_n_steps=log_freq, devices=devices, accelerator=accelerator, callbacks=[early_stop_callback])

    # strategy="ddp"   
    
    if debug:
        pdb.set_trace(header="After trainer instantation")
    
    # tune to find the learning rate
    #trainer.tune(gnn_model,datamodule=data_module)
    
    # we can resume from a checkpoint using trainer.fit(ckpth_path="some/path/to/my_checkpoint.ckpt")
    trainer.fit(gnn_model, datamodule=data_module, ckpt_path=args.checkpoint)
    
    if debug:
        pdb.set_trace(header="After trainer fit")
    
    trainer.validate(gnn_model, datamodule=data_module)
    
    wandb.finish()
    
    
    
    
def filter_target(target_names:List[str], target:str)-> Callable[[Data],Data]:
    """ Transform to be given to a Dataset, has the effect of filtering out all irelevant targets in the Data objects in the dataset at runtime
    Example: for BACE, target_names=['Class', 'PIC50'], we want to train a classifier => target='Class'
    """
    target_idx = target_names.index(target)

    return partial(filter_target_with_idx, target_idx=target_idx)

def filter_target_with_idx(graph:Data, target_idx:int) -> Data:
    new_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, pos=graph.pos ,y=graph.y[:,target_idx:target_idx+1], z=graph.z, name=graph.name, idx=graph.idx) 

    return new_graph



if __name__=="__main__":
    main()
