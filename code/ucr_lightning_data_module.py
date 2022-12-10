import pytorch_lightning as pl

from torch_geometric.data import Dataset
from torch_geometric.loader import  DataLoader
import math


class UcrDataModule(pl.LightningDataModule):
    """ Pytorch Ligthning Data Module wrapper around Smiles Dataset to ensure reproducible and easy splitting of the dataset"""
    
    def __init__(self, dataset:Dataset, seed, train_frac=0.6, valid_frac=0.1, test_frac=0.3, batch_size=32, test_dataset:Dataset = None) -> None:
        super().__init__()
    
        fracs= [train_frac,valid_frac,test_frac]

        if not(math.isclose(sum(fracs), 1) and sum(fracs) <= 1):
            raise ValueError("invalid train_val_test split, fractions must add up to 1")
        
        if test_dataset is not None:
            # normalizing fractions if test_frac is not useful because we are provided the test dataset
            norm_factor = train_frac + valid_frac
            train_frac = train_frac/norm_factor
            valid_frac= valid_frac/norm_factor

        #self.lengths = self.lengths_from_frac(fracs=fracs)
        self.batch_size = batch_size        
        self.dataset = dataset
        
        # seeding
        pl.seed_everything(seed=seed, workers=True)
        
        
        # splitting the dataset
        
        self.dataset = self.dataset.shuffle()
        num_samples = len(self.dataset)
        
        num_train = math.floor(num_samples*train_frac)
        self.train_data = self.dataset[:num_train]
        
        if test_dataset is not None:
            self.valid_data = self.dataset[:num_train]
            self.test_data = test_dataset
        else:
            num_valid = math.floor(num_samples*valid_frac)
            self.valid_data = self.dataset[num_train : num_train + num_valid]
            self.test_data = self.dataset[num_train + num_valid:]
    
        
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
    
    
    
