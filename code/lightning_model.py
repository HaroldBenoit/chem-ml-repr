import time
import numpy as np
import torch
from torch.nn import Dropout, Linear, ReLU, PReLU
import torch_geometric
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GeneralConv, Sequential, global_add_pool

from typing import Any


# these imports are only used in the Lighning version
import pytorch_lightning as pl
import torch.nn.functional as F




class LightningClassicGNN(pl.LightningModule):
    #! following https://graphneural.network/models/#generalgnn implementation
    
    def __init__(self, classification = True, num_hidden_features=256, dropout_p=0.0, learning_rate=0.01, **kwargs: Any) -> None:
        super(LightningClassicGNN,self).__init__()
        
        if "num_node_features" in kwargs:
            self.num_node_features = kwargs["num_node_features"]
        else:
            raise Exception("num_node_features not defined")
        
        if "num_edge_features" in kwargs:
            self.num_edge_features = kwargs["num_edge_features"]
        else:
            raise Exception("num_edge_features not defined")
        
        
        if classification:
            if "num_classes" in kwargs:
                self.num_classes = kwargs["num_classes"]
            else:
                raise ValueError("model set to classification but num_classes not given")
            
        else:
            
            if "output_dim" in kwargs:
                self.output_dim = kwargs["output_dim"]
            else:
                raise ValueError("model set to regression but num_classes not given")
            

        #! add batch norm
        #! add distinction between classification and regression
        
        self.classification = classification        
        self.hidden = num_hidden_features
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        
        # this is function header definition i.e. input args and return: "arg1, arg2 -> return_type"
        
        layers = [
            (GeneralConv(in_channels=self.num_node_features, in_edge_channels=self.num_edge_features,out_channels=self.hidden), "x, edge_index -> x1"),
            (ReLU(), "x1 -> x1a"),
            (Dropout(p = self.dropout_p), "x1a -> x1d"),
            (GeneralConv(self.hidden, self.hidden), "x1d, edge_index -> x2"), 
            (ReLU(), "x2 -> x2a"),                                        
            (Dropout(p=self.dropout_p), "x2a -> x2d"),                                
            (GeneralConv(self.hidden, self.hidden), "x2d, edge_index -> x3"),  
            (ReLU(), "x3 -> x3a"),                                       
            (Dropout(p=self.dropout_p), "x3a -> x3d"),                             
            (GeneralConv(self.hidden, self.hidden), "x3d, edge_index -> x4"), 
            (ReLU(), "x4 -> x4a"),                                      
            (Dropout(p=self.dropout_p), "x4a -> x4d"),                               
            (GeneralConv(self.hidden, self.hidden), "x4d, edge_index -> x5"), 
            (ReLU(), "x5 -> x5a"),                                  
            (Dropout(p=self.dropout_p), "x5a -> x5d"),
            (global_add_pool, "x5d, batch_index -> x6" ),

        ]
        
        
        # changing end layer depending on whether we are classfiying or regressing
        if self.classification:
            layers.append((Linear(self.hidden, self.num_classes), "x6 -> x_out"))
        else: 
            layers.append((Linear(self.hidden, self.output_dim), "x6 -> x_out"))
        
        
        
        self.model = Sequential("x, edge_index, batch_index",layers) 
        
        
        
    def forward(self, x, edge_index, batch_index):
        x_out = self.model(x,edge_index, batch_index)
        
        return x_out
    
    
    def training_step(self, batch, batch_index):
        x, edge_index = batch.x , batch.edge_index
        batch_index = batch.batch
        
        x_out = self.forward(x, edge_index, batch_index)
        
        if self.classification:
            loss = F.cross_entropy(x_out, batch.y)
        
            # metrics here
            pred = x_out.argmax(-1)
            label = batch.y
            accuracy = (pred ==label).sum() / pred.shape[0]

            self.log("loss/train", loss)
            self.log("accuracy/train", accuracy)
        else:
            loss= F.l1_loss(x_out,batch.y)
            
        
        return loss
    
    def validation_step(self, batch, batch_index):
        
        x, edge_index = batch.x , batch.edge_index
        batch_index = batch.batch
        
        x_out = self.forward(x, edge_index, batch_index)

        loss = F.cross_entropy(x_out, batch.y)

        pred = x_out.argmax(-1)

        return x_out, pred, batch.y
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        return optimizer
    