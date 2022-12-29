import torch
from torch.nn import Dropout, Linear,PReLU

from torch_geometric.nn import GeneralConv, Sequential, global_add_pool, BatchNorm
from sklearn.metrics import roc_auc_score


from typing import Any


# these imports are only used in the Lighning version
import pytorch_lightning as pl
import torch.nn.functional as F

import pdb




class LightningClassicGNN(pl.LightningModule):
    """ GNN model wrapped in a Pytorch Lightning Module 
        #! following https://graphneural.network/models/#generalgnn implementation

    """
    
    def __init__(self, seed:int, classification = True, num_hidden_features=256, dropout_p=0.0, learning_rate=1e-3, num_message_passing_layers=4 , val_check_interval=1.0, **kwargs: Any) -> None:
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
        self.seed = seed
        self.val_check_interval = val_check_interval
        pl.seed_everything(seed=seed, workers=True)

        
        # how to build layers: function header definition i.e. input args and return: "arg1, arg2 -> return_type"
        
        # first initial layer
        layers=[(GeneralConv(in_channels=self.num_node_features, in_edge_channels=self.num_edge_features,out_channels=self.hidden), "x, edge_index-> x0"),
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
        return self.forward_step(batch=batch, batch_index=batch_index, is_train=True)

    
    def validation_step(self, batch, batch_index):
        
        return self.forward_step(batch=batch, batch_index=batch_index, is_train=False)
    
    
    def forward_step(self,batch,batch_index, is_train: bool):
            x, edge_index = batch.x , batch.edge_index
            batch_index = batch.batch
            batch_size= len(batch)
            #print("x ", x)
            #print("edge_index: ", edge_index)
            #print("batch_index: " ,batch_index)
            
            x_out = self.forward(x, edge_index, batch_index)
            
            suffix= "train" if is_train else "valid"


            if self.classification:

                batch_y = torch.squeeze(batch.y,1)
                loss = F.cross_entropy(x_out, batch_y.long())

                # metrics here
                #x_out_cpu = x_out.detach().cpu()
                #print("X_out", x_out_cpu)
                prob = F.softmax(input=x_out, dim=1)[:,1]
                batch_y_cpu=batch_y.detach().cpu().numpy()
                prob_cpu = prob.detach().cpu().numpy()
                #print("batch_y ",batch_y_cpu)
                #print("prob ", prob_cpu)
                auc= roc_auc_score(y_true= batch_y_cpu, y_score= prob_cpu)
                
                #print("Auc", auc)

                pred = x_out.argmax(-1)
                label = batch.y
                accuracy = (pred ==label).sum() / pred.shape[0]

                self.log(f"loss/{suffix}", loss, batch_size=batch_size, on_epoch=True)
                self.log(f"auc/{suffix}", float(auc) ,batch_size=batch_size, on_epoch=True)
                self.log(f"accuracy/{suffix}",accuracy, batch_size=batch_size, on_epoch=True)
            else:
                loss= F.l1_loss(x_out,batch.y)
                self.log(f"loss/{suffix}", loss, batch_size=batch_size, on_epoch=True, sync_dist= suffix==valid)


            return loss
        
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1)
        
        lr_scheduler_config={
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": scheduler,
                "monitor": "loss/val",
                "frequency":self.val_check_interval, ##!TODO could be equal to val_check_interval from training.py
            }
        }  
        return lr_scheduler_config
    