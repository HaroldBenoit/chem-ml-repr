from ucr_lightning_data_module import UcrDataModule
from lightning_model import LightningClassicGNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
# making sure we are as determinstic as possibe
#torch.use_deterministic_algorithms(True)
import numpy as np
from time import gmtime, strftime   
from torch_geometric.data import Data
from typing import List, Callable
from functools import partial
from torch_geometric.transforms import Compose
from distance import Distance
import wandb
import pdb
import argparse
import datasets_classes
import os 
import pandas as pd

def main():
    
    parser = argparse.ArgumentParser(prog="Training", description="Training pipeline")

    
    ##dataset
    parser.add_argument('--root', required=True, help="Root path where the dataset is stored or to be stored after processing")
    parser.add_argument('--dataset', required=True, help=f"Dataset name. Available datasets are {list(datasets_classes.dataset_dict.keys())}")
    parser.add_argument('--target', required=True, help="Target name i.e. predicted value in dataset")
    parser.add_argument('--seed', default=42, type=int, help="Seed that dictates dataset splitting")
    parser.add_argument('--total_frac', default=1.0, type=float, help="Total fraction of the dataset to use, can be useful to set < 1.0 for experimentation")
    parser.add_argument('--train_frac', default=0.6, type=float, help="Fraction of the dataset to use as training set")
    parser.add_argument('--valid_frac', default=0.2, type=float, help="Fraction of the dataset to use as validation set")
    parser.add_argument('--test_frac', default=0.2, type=float, help="Fraction of the dataset to use as testing set")
    parser.add_argument('--scaffold', action='store_true', help="If flag specified, use scaffold splitting (only available for molecular datasets)")


    #parser.add_argument('--weighted', action="store_true", help="If flag specified, make the edge distances weighted by atomic radius")
    #parser.add_argument('--no_distance', action='store_true',help="If flag specified, don't compute distance")
    #parser.add_argument('--dist_present', action="store_true", help="If flag specified, dist has been computed")
    
    
    parser.add_argument('--debug', action='store_true', help="If flag specified, activate breakpoints in the script")
    parser.add_argument('--no_log', action="store_true", help="If flag specified, no logging is done")
    parser.add_argument('--cluster', action='store_true' ,help="If flag specified, we are training on the cluster")
    parser.add_argument('--model_checkpoint', help="Path to the checkpoint of the model (ends with .ckpt). Defaults to None")
    parser.add_argument('--training_checkpoint', help="Path to the checkpoint of the model and its training (ends with .ckpt). Defaults to None")
    parser.add_argument('--run_name', default=None, help="Run name for logging purposes")
    parser.add_argument('--project_name', default=None, help="Project name for logging purposes")
    parser.add_argument('--results', action='store_true', help="If flag specified, we want to have the final results for this model")
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


    ## stratified if classification but scaffold has priority over it
    use_stratified = dataset_class.is_classification[target] and not(args.scaffold)
    
        
    
    dataset = dataset_class(root=root, add_hydrogen=args.hydrogen,transform=transforms)
    
    target_idx = dataset_class.target_names.index(target)
    # from torch dataset, create lightning data module to make sure training splits are always done the same ways
    data_module = UcrDataModule(dataset=dataset, seed=seed, batch_size=args.batch_size, total_frac = args.total_frac, stratified=use_stratified, scaffold_split=args.scaffold, target_idx=target_idx, train_frac=args.train_frac, valid_frac=args.valid_frac, test_frac=args.test_frac)

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
    
    if args.model_checkpoint is None:
        gnn_model = LightningClassicGNN(seed=seed, classification=classification, output_dim=output_dim, dropout_p=dropout_p,
                                    num_hidden_features=num_hidden_features,  num_node_features=num_node_features, 
                                    num_edge_features=num_edge_features, num_message_passing_layers=args.num_message_layers, 
                                    total_steps=total_steps)
    else:
        gnn_model = LightningClassicGNN.load_from_checkpoint(args.model_checkpoint)
    
    
    ## WANDB
      
    if args.project_name is None:
        project=f"{args.dataset}-project-post2"
    else:
        project = args.project_name
        
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
    patience = num_epochs //2
    early_stop_callback = EarlyStopping(monitor="loss/valid", mode="min", patience=patience, min_delta=0.00)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    callbacks=[]
        
    if num_epochs > 10:
        callbacks.append(early_stop_callback)
    if not(args.no_log):
        callbacks.append(lr_monitor)
    
    if args.results:
        save_dir = "../experiments_results"
        csv_log_name = f"{project}_{run_name}"
        version = "final_results_seed_{args.seed}"
        logger = CSVLogger(save_dir=save_dir, name=csv_log_name, version= version)
    else:
        logger=wandb_logger
        

    
    trainer = pl.Trainer(logger=logger, deterministic=False, default_root_dir="../training_artifacts/", precision=32,
	 strategy=strategy,max_epochs=num_epochs ,log_every_n_steps=log_freq, devices=devices, accelerator=accelerator, callbacks=callbacks, fast_dev_run=False, val_check_interval=args.val_check_interval)

    # strategy="ddp"   
    
    if debug:
        pdb.set_trace(header="After trainer instantation")
    
    # tune to find the learning rate
    #trainer.tune(gnn_model,datamodule=data_module)
    
    # we can resume from a checkpoint using trainer.fit(ckpth_path="some/path/to/my_checkpoint.ckpt")
    trainer.fit(gnn_model, datamodule=data_module, ckpt_path=args.training_checkpoint)
    
    if debug:
        pdb.set_trace(header="After trainer fit")
    
    
    ## logging the best validation metric into the global results dataframe
    
    if args.results:
        filename = "metrics.csv"
        path = os.path.join(save_dir, csv_log_name, version, filename)
        curr_res = pd.read_csv(path)
        ## need to differentiate between classification and regression
        
        if classification:
            metric_name = "auc"
            metric_value = curr_res['auc/valid'].max()
        else:
            metric_name = "mae"
            metric_value = curr_res["loss/valid"].min()
            
        # df schema
        # "dataset":[],
        # "target":[],
        # "seed":[],
        # "time":[],
        # "metric_name":[],
        # "metric_value":[]
        
        time = pd.Timestamp.now()
        
        columns = ["dataset","target","seed","time", "epochs", "train_frac", "valid_frac", "metric_name", "metric_value"]
        new_row = pd.DataFrame([[args.dataset, args.target, args.seed, time, num_epochs, args.train_frac, args.valid_frac, metric_name, metric_value]],columns=columns)
        new_row = new_row.set_index('dataset')
        
        global_res_path = "../experiments_results/global_results_random_generation.csv" if not(args.boolean) else "../experiments_results/global_results_boolean.csv"
        if os.path.exists(global_res_path):
            global_res = pd.read_csv(global_res_path, index_col='dataset')
        else:
            global_res = pd.DataFrame([],columns=columns)
            global_res = global_res.set_index('dataset')
            
        global_res = pd.concat([global_res, new_row])

        global_res.to_csv(global_res_path)
        
        

            
        
    ## for validation, it is important to make sure that we're running everything on the same GPU not to have duplicate batches
    #trainer = pl.Trainer( deterministic=False, default_root_dir="../training_artifacts_test/", precision=32, accelerator='gpu', devices=1,  logger = False)
    #trainer = pl.Trainer(logger=logger)
    #
    #val_metrics = trainer.validate(gnn_model, datamodule=data_module)
    #print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    #print("VAL METRICS",val_metrics)
    
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
