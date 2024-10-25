from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from ddp_util import all_gather
from logger import Logger
import os
import datetime
class Trainer(ABC):
    
    def __init__(self, 
        model           : nn.Module,
        loss_fn         : nn.Module,
        optimizer       : torch.optim.Optimizer,
        scheduler   ,
        train_loader    : torch.utils.data.DataLoader,
        valid_loader    : torch.utils.data.DataLoader,
        logger          : Logger,
        path_save_dir   : str,
        device          : int
    ):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.logger = logger
        
        self.path_save_dir = path_save_dir
        
        self.device = device
        
    @abstractmethod
    def train(self):
        ...      
    
    @abstractmethod
    def valid(self) -> float:
        """
        Evaluate model on validation set
        
        Returns:
            float: avg err
        """
        ...
    
    def calculate_RMSE(self, infers: torch.Tensor, labels: torch.Tensor) -> float:
        infers, labels = infers.float(), labels.float()
        mse = torch.mean((labels - infers) ** 2)
        return torch.sqrt(mse).numpy()
        
    
    def run(
        self,
        max_epoch:int,
        save_parameter:bool, 
        ):
        
        best_err = 100.
        for epoch in range(1, max_epoch + 1):
            
            self.logger.print(f'epoch: {epoch}')
            self.train()
            self.scheduler.step()
            
            # -------------------- evaluation
            err = self.valid()
            self.logger.print(f'err: {err}')
            self.logger.wandbLog(
                    {
                        'err' : err, 
                        'learning_rate': self.scheduler.get_last_lr()[0] , 
                        'epoch' : epoch
                    }
                )
                
            if err < best_err:

                best_err = err
                self.logger.wandbLog({'err' : err, 'epoch' : epoch})
                    
                if self.device == 0 and save_parameter:
                    self.save_model(
                            error=err
                        )
        
                    
                
    def save_model(
        self,
        error:float
        ):
        if self.device != 0:
            return
        
        directory = datetime.date.today().strftime("%Y-%m-%d")
        directory = os.path.join(self.path_save_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        save_name = str(error)+".pt"
        full_path = os.path.join(directory, save_name)
        
        self.logger.print("saving... path: " + full_path)
        torch.save(self.model.state_dict(), full_path)
    
