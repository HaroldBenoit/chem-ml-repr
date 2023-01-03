import pytorch_lightning as pl

from torch_geometric.loader import  DataLoader
import math
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from typing import Union
from in_mem_ucr_dataset import InMemoryUcrDataset
from ucr_dataset import UcrDataset

class UcrDataModule(pl.LightningDataModule):
    """ Pytorch Ligthning Data Module wrapper around Smiles Dataset to ensure reproducible and easy splitting of the dataset"""
    
    def __init__(self, dataset:Union[InMemoryUcrDataset, UcrDataset], seed, stratified = False, train_frac=0.6, valid_frac=0.1, test_frac=0.3, batch_size=32, total_frac=1.0) -> None:
        super().__init__()
    
        fracs= [train_frac,valid_frac,test_frac]

        if not(math.isclose(sum(fracs), 1) and sum(fracs) <= 1):
            raise ValueError("invalid train_val_test split, fractions must add up to 1")
        
        #if test_dataset is not None:
        #    # normalizing fractions if test_frac is not useful because we are provided the test dataset
        #    norm_factor = train_frac + valid_frac
        #    train_frac = train_frac/norm_factor
        #    valid_frac= valid_frac/norm_factor

        self.batch_size = batch_size        
        self.dataset = dataset
        
        # seeding
        pl.seed_everything(seed=seed, workers=True)
        
        if stratified:
            y= dataset.y.numpy()
            y_len = y.shape[0]
            ## trimming the dataset by total_frac
            y = y[:int(y_len*total_frac)]
            
            x = np.arange(y.shape[0])

            #first split to get training data
            sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_frac+test_frac, random_state=seed)

            train_index , val_test_index = list(sss.split(x,y))[0]

            self.train_data = self.dataset[train_index]
            
            # second split to get val and test data
            new_y = y[val_test_index]
            new_x = np.arange(new_y.shape[0])
            new_test_frac = test_frac/(valid_frac + test_frac)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=new_test_frac, random_state=seed)
            
            new_val_index, new_test_index = list(sss.split(new_x, new_y))[0]
            
            val_index = val_test_index[new_val_index]
            test_index = val_test_index[new_test_index]
            
            self.valid_data = self.dataset[val_index]
            self.test_data = self.dataset[test_index]

        else:
            # splitting the dataset
            self.dataset = self.dataset.shuffle()
            ## simple way to trim the dataset size is to multiply by total_frac 
            num_samples = math.floor(len(self.dataset)*total_frac)

            num_train = math.floor(num_samples*train_frac)
            self.train_data = self.dataset[:num_train]


            num_valid = math.floor(num_samples*valid_frac)
            self.valid_data = self.dataset[num_train : num_train + num_valid]
            self.test_data = self.dataset[num_train + num_valid:num_samples]
        
    
        
        # grabbing node and edge features
        graph = self.train_data[0]
        self.num_node_features = graph.x.shape[1]
        self.num_edge_features= graph.edge_attr.shape[1]
        
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=4)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, batch_size=self.batch_size, num_workers=4)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)
    
    
    
