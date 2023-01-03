from ucr_lightning_data_module import UcrDataModule
from lightning_model import LightningClassicGNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
# making sure we are as determinstic as possibe
#torch.use_deterministic_algorithms(True)
import numpy as np

from torch_geometric.data import Data
from typing import List, Callable
from functools import partial
from torch_geometric.transforms import Compose
from distance import Distance
import wandb
import pdb
import argparse
import datasets_classes

def main():
    
    parser = argparse.ArgumentParser(prog="Training", description="Training pipeline")

    
    ##dataset
    parser.add_argument('--root', required=True, help="Root path where the dataset is stored or to be stored after processing")
    parser.add_argument('--dataset', required=True, help=f"Dataset name. Available datasets are {list(datasets_classes.dataset_dict.keys())}")
    parser.add_argument('--target', required=True, help="Target name i.e. predicted value in dataset")
    parser.add_argument('--seed', default=42, type=int, help="Seed that dictates dataset splitting")
    parser.add_argument('--total_frac', default=1.0, type=float, help="Total fraction of the dataset to use, can be useful to set < 1.0 for experimentation")
    #parser.add_argument('--weighted', action="store_true", help="If flag specified, make the edge distances weighted by atomic radius")
    #parser.add_argument('--no_distance', action='store_true',help="If flag specified, don't compute distance")
    #parser.add_argument('--dist_present', action="store_true", help="If flag specified, dist has been computed")
    
    
    parser.add_argument('--debug', action='store_true', help="If flag specified, activate breakpoints in the script")
    parser.add_argument('--no_log', action="store_true", help="If flag specified, no logging is done")
    parser.add_argument('--cluster', action='store_true' ,help="If flag specified, we are training on the cluster")
    parser.add_argument('--checkpoint', help="Path to the checkpoint of the model (ends with .ckpt). Defaults to None")
    parser.add_argument('--run_name', default=None, help="Run name for logging purposes")
    ##representation
    parser.add_argument('--hydrogen', action='store_true' ,help="If flag specified, we are using explicit hydrogens")
    parser.add_argument('--boolean', action='store_true', help="If flag specified, we are also using boolean features in the node features")


    ##training
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_hidden', default=256, type=int)
    parser.add_argument('--num_message_layers', default=4, type=int)
    parser.add_argument('--val_check_interval', default=1.0, type=float,help="How often within one training epoch to check the validation set. Can specify as float")
    
    
    print("\nIf you're dealing with materials data, it is a good idea to reduce the number of epochs (4), reduce batch size (4), lower val_check_interval (0.25) and maybe consider subsampling (total_frac < 1.0)s\n")
    
    
    args = parser.parse_args()
    debug= args.debug
    ## dataset
    root= args.root
    dataset=args.dataset
    target=args.target
    #dist_present = args.dist_present
    #no_distance = args.no_distance

    seed=args.seed
    

    ## pytorch lighting takes of seeding everything
    pl.seed_everything(seed=seed, workers=True)
    
    ## training

    num_epochs=args.epochs
    # (int) log things every N batches
    log_freq=1
    accelerator="gpu"
	#good devices are 0,1,2,3 on the cluster
    if args.cluster:
        devices = [0,1,2]
    else:
        devices = [0]
    
    
    if debug:
        pdb.set_trace(header="Before dataset transform")
    

    
    
    ## DATASET
    ## getting the correct dataset
    dataset_class = datasets_classes.dataset_dict[dataset]
    
    #weighted = args.weighted
    #atom_number_to_radius = None if not(weighted) else torch.load("../important_data/atom_number_to_radius.pt")
    
    
    # filtering out irrelevant target and removing or not boolean features
    if args.boolean:
        transforms = filter_target(target_names=dataset_class.target_names, target=target)
    else:
        transforms = Compose([filter_target(target_names=dataset_class.target_names, target=target), filter_boolean_features])


    use_stratified = dataset_class.is_classification[target]
        
    
    dataset = dataset_class(root=root, add_hydrogen=args.hydrogen,transform=transforms)
    
    target_idx = dataset_class.target_names.index(target)
    # from torch dataset, create lightning data module to make sure training splits are always done the same ways
    data_module = UcrDataModule(dataset=dataset, seed=seed, batch_size=args.batch_size, total_frac = args.total_frac, stratified=use_stratified, target_idx=target_idx)

    if debug:
        pdb.set_trace(header="After dataset transform")
    

    
    ## MODEL
    num_hidden_features=args.num_hidden
    dropout_p = 0.0
    classification= dataset_class.is_classification[target]
    output_dim = 2 if classification else 1
    
    num_node_features = data_module.num_node_features
    num_edge_features= data_module.num_edge_features
      
    ## need total_steps for lr_scheduler
    total_steps = int(len(data_module.train_dataloader()) / args.batch_size) * num_epochs * 2
    
    gnn_model = LightningClassicGNN(seed=seed, classification=classification, output_dim=output_dim, dropout_p=dropout_p,
                                    num_hidden_features=num_hidden_features,  num_node_features=num_node_features, 
                                    num_edge_features=num_edge_features, num_message_passing_layers=args.num_message_layers, 
                                    total_steps=total_steps)
    
    
    ## WANDB
      
    project=f"{args.dataset}-project-post2"
    run_name=f"target_{target}" 

    if args.hydrogen:
        run_name=f"{run_name}_hydrogen"
        
        
    if args.boolean:
        run_name=f"{run_name}_boolean"

        
    if args.run_name is not None:
        run_name = args.run_name
    
    if args.no_log:
        wandb_logger=False
    else:
        #docs: https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.WandbLogger.html#pytorch_lightning.loggers.WandbLogger
        ## maybe put id=run_name
        wandb_logger = WandbLogger(save_dir="../training_artifacts/", log_model=True, project=project, name=run_name)
        ## log histograms of gradients and parameters
        wandb_logger.watch(gnn_model, log_freq=log_freq)
        
        
    ## TRAINER
    
    strategy = "ddp" if args.cluster else None
    ## creating early stop callback to ensure we don't overfit
    if num_epochs > 10:
        patience = num_epochs //2
    else:
        patience = num_epochs
    early_stop_callback = EarlyStopping(monitor="loss/valid", mode="min", patience=patience, min_delta=0.00)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks = [early_stop_callback, lr_monitor] if not(args.no_log) else [early_stop_callback]
    
    trainer = pl.Trainer(logger=wandb_logger, deterministic=False, default_root_dir="../training_artifacts/", precision=32,
	 strategy=strategy,max_epochs=num_epochs ,log_every_n_steps=log_freq, devices=devices, accelerator=accelerator, callbacks=callbacks, fast_dev_run=False, val_check_interval=args.val_check_interval)

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
    if target not in target_names:
        raise ValueError(f"Given target {target} is not present in targets {target_names}")
    
    target_idx = target_names.index(target)

    return partial(filter_target_with_idx, target_idx=target_idx)

def filter_target_with_idx(graph:Data, target_idx:int) -> Data:
    new_graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, pos=graph.pos ,y=graph.y[:,target_idx:target_idx+1], z=graph.z, name=graph.name, idx=graph.idx) 

    return new_graph


def filter_boolean_features(graph:Data) -> Data:
    ## we remove all the "is_*" features from the node features
    # we also remove useless parts to make the representation use less memory
    new_graph = Data(x=graph.x[:,:10], edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y) 

    return new_graph




if __name__=="__main__":
    main()
