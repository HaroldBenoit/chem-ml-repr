import time
import numpy as np
import torch
from torch.nn import Dropout, Linear, ReLU, PReLU
import torch_geometric
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GeneralConv, Sequential, global_add_pool, BatchNorm

from typing import Any


# these imports are only used in the Lighning version
import pytorch_lightning as pl
import torch.nn.functional as F




class LightningClassicGNN(pl.LightningModule):
    """ GNN model wrapped in a Pytorch Lightning Module 
        #! following https://graphneural.network/models/#generalgnn implementation

    """
    
    def __init__(self, classification = True, num_hidden_features=256, dropout_p=0.0, learning_rate=0.01, num_message_passing_layers=4, **kwargs: Any) -> None:
        super(LightningClassicGNN,self).__init__()
        self.save_hyperparameters()
        
        if "num_node_features" in kwargs:
            self.num_node_features = kwargs["num_node_features"]
        else:
            raise Exception("num_node_features not defined")
        
        if "num_edge_features" in kwargs:
            self.num_edge_features = kwargs["num_edge_features"]
        else:
            raise Exception("num_edge_features not defined")
        
    
        if "output_dim" in kwargs:
            self.output_dim = kwargs["output_dim"]
        else:
            raise ValueError("output_dim not given")
                    
        self.classification = classification        
        self.hidden = num_hidden_features
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.num_message_passing_layers= num_message_passing_layers
        
        # how to build layers: function header definition i.e. input args and return: "arg1, arg2 -> return_type"
        
        # first initial layer
        layers=[(GeneralConv(in_channels=self.num_node_features, in_edge_channels=self.num_edge_features,out_channels=self.hidden), "x, edge_index -> x0"),
                (BatchNorm(in_channels=self.hidden), "x0 -> x0a"),
                (PReLU(), "x0a -> x0b"),
                (Dropout(p = self.dropout_p), "x0b -> x0c"),]
        
        #other message passing layers
        for i in range(num_message_passing_layers):
            layers.append((GeneralConv(self.hidden, self.hidden), f"x{i}c, edge_index -> x{i+1}"))
            layers.append((BatchNorm(in_channels=self.hidden), f"x{i+1} -> x{i+1}a"))
            layers.append((PReLU(), f"x{i+1}a -> x{i+1}b"))
            layers.append((Dropout(p = self.dropout_p), f"x{i+1}b -> x{i+1}c"))
            
        last_i = num_message_passing_layers -1
        layers.append((global_add_pool, f"x{last_i +1}c, batch_index -> x{last_i+2}"))
        layers.append((Linear(self.hidden,self.hidden), f"x{last_i +2} -> x{last_i +3}"))
        layers.append((Linear(self.hidden, self.output_dim), f"x{last_i +3} -> x_out"))
    
        
        self.model = Sequential("x, edge_index, batch_index",layers) 
        


        
        
    def forward(self, x, edge_index, batch_index):
        x_out = self.model(x,edge_index, batch_index)
        
        return x_out
    
    
    def training_step(self, batch, batch_index):
        x, edge_index = batch.x , batch.edge_index
        batch_index = batch.batch
        batch_size= len(batch)
        x_out = self.forward(x, edge_index, batch_index)
        
        if self.classification:

            loss = F.cross_entropy(x_out, torch.squeeze(batch.y,1).long())
        
            # metrics here
            pred = x_out.argmax(-1)
            label = batch.y
            accuracy = (pred ==label).sum() / pred.shape[0]

            self.log("loss/train", loss, batch_size=batch_size)
            self.log("accuracy/train", accuracy, batch_size=batch_size)
        else:
            loss= F.l1_loss(x_out,batch.y)
            self.log("loss/train", loss, batch_size=batch_size)
            
        
        return loss
    
    def validation_step(self, batch, batch_index):
        
        x, edge_index = batch.x , batch.edge_index
        batch_index = batch.batch
        batch_size= len(batch)
        x_out = self.forward(x, edge_index, batch_index)
        
        if self.classification:
            
            loss = F.cross_entropy(x_out, torch.squeeze(batch.y,1).long())
        
            # metrics here
            pred = x_out.argmax(-1)
            label = batch.y
            accuracy = (pred ==label).sum() / pred.shape[0]

            self.log("loss/valid", loss, batch_size=batch_size)
            self.log("accuracy/valid", accuracy,batch_size=batch_size)
            
            return x_out, pred, batch.y
        else:
            
            loss= F.l1_loss(x_out,batch.y)
            self.log("loss/valid", loss, batch_size=batch_size)
            return x_out, batch.y
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    