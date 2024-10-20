import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from ddp_util import all_gather
from logger import Logger
import os

class Trainer:
    
    def __init__(self, 
        model       : nn.Module,
        loss_fn     : nn.Module,
        optimizer   : torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        test_loader : torch.utils.data.DataLoader,
        logger      : Logger,
        device      : int
    ):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        self.logger = logger
        
        self.device = device
        
        
    def train(self):
        
        self.model.train()
        self.loss_fn.train()
        
        iter_count = 0
        loss_sum = 0
        num_item_train = len(self.train_loader)
        pbar = tqdm(self.train_loader)
        for product_id, rating, ref_ids, ref_ratings in pbar:
            
            self.optimizer.zero_grad()
            
            product_id  = product_id.to(self.device)
            rating      = rating.to(self.device)
            ref_ids     = ref_ids.to(self.device)
            ref_ratings = ref_ratings.to(self.device)
            
            assert len(product_id.shape)  == 2, "batch of product_id: [batch, 1]"
            assert len(rating.shape)      == 2, "batch of rating: [batch, 1]"
            assert len(ref_ids.shape)     == 2, "batch of ref_ids: [batch, max num of reference]"
            assert len(ref_ratings.shape) == 2, "batch of ref_ratings: [batch, max num of reference]"
            
            ###################### forward and backward
            infer = self.model(product_id, ref_ids, ref_ratings)
            loss = self.loss_fn(infer, rating)
            
            loss.backward()
            self.optimizer.step()

            ###################### logging
            loss = loss.detach()
            iter_count += 1
            loss_sum += loss
            
            pbar.set_description(f'loss: {loss}')
            
            if num_item_train * 0.02 <= iter_count:
                self.logger.wandbLog({'Loss' : loss_sum / float(iter_count)})
                loss_sum = 0
                iter_count = 0
                
                
    def valid(self) -> float:
        """
        Evaluate model on validation set
        
        Returns:
            float: avg err
        """
        
        self.model.eval()
        
        infer_list = []
        label_list = []
        
        pbar = tqdm(self.test_loader, 'evaluation')
        with torch.no_grad():
            
            for product_id, rating, ref_ids, ref_ratings in pbar:
            
                product_id  = product_id.to(self.device)
                rating      = rating.to(self.device)
                ref_ids     = ref_ids.to(self.device)
                ref_ratings = ref_ratings.to(self.device)
                
                y = self.model(product_id, ref_ids, ref_ratings)

                infer_list.append(y.cpu().squeeze())
                label_list.append(rating.cpu().squeeze())

        infer_list = torch.cat(infer_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
                    
        infer_list = all_gather(infer_list)
        label_list = all_gather(label_list)
        
        infer_list = torch.cat(infer_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        
        rmse = self.calculate_RMSE(infer_list, label_list)
        
        
        return rmse
            
    def calculate_RMSE(self, infers: torch.Tensor, labels: torch.Tensor) -> float:
        # RMSE
        
        n = len(infers)
        x = infers - labels
        x = (x.square()/n).sqrt()
        x = x.mean()
        return x.numpy().astype(float)
        
    
    def run(self, max_epoch, path_parameter_storage):
        
        best_err = 100.
        for epoch in range(1, max_epoch + 1):
            
            self.logger.print(f'epoch: {epoch}')
            self.train()
            #scheduler.step()
            
            # -------------------- evaluation
            if epoch % 5 == 1 or epoch == max_epoch:
                
                err = self.valid()
                self.logger.print(f'err: {err}')
                self.logger.wandbLog({'err' : err, 'epoch' : epoch})
                
                if err < best_err:

                    best_err = err
                    self.logger.wandbLog({'err' : err, 'epoch' : epoch})
                    
                    if self.device == 0:
                        self.save_model(
                            path_parameter_storage=path_parameter_storage,
                            save_name=f"{str(err)[0:6]}.pt"
                        )
                
    def save_model(
        self, 
        path_parameter_storage, 
        save_name
        ):
        
        if self.device != 0:
            return
        
        full_path = os.path.join(path_parameter_storage, save_name)
        self.logger.print("saving... path: " + full_path)
        
        torch.save(self.model.state_dict(), full_path)
    
