import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from ddp_util import all_gather
from logger import Logger
from trainers import base

class Trainer(base.Trainer):
    
    def __init__(self, 
        model       : nn.Module,
        loss_fn     : nn.Module,
        optimizer   : torch.optim.Optimizer,
        scheduler   ,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        logger      : Logger,
        path_save_dir: str,
        device      : int
    ):
        
        super(Trainer, self).__init__(
            model = model,
            loss_fn= loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            logger=logger,
            path_save_dir=path_save_dir,
            device=device
        )
        
        
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
            rating      = rating.squeeze().type(torch.int64).to(self.device) - 1
            ref_ids     = ref_ids.to(self.device)
            ref_ratings = ref_ratings.to(self.device)
            
            assert len(product_id.shape)  == 2, "batch of product_id: [batch, 1]"
            assert len(rating.shape)      == 1, "batch of rating: [batch]"
            assert len(ref_ids.shape)     == 2, "batch of ref_ids: [batch, max num of reference]"
            assert len(ref_ratings.shape) == 2, "batch of ref_ratings: [batch, max num of reference]"
            
            ###################### forward and backward
            infer = self.model(product_id, ref_ids, ref_ratings)
            #print(infer.shape)
            #print(rating.shape)
            loss = self.loss_fn(infer, rating)
            
            loss.backward()
            self.optimizer.step()

            ###################### logging
            loss = loss.detach()
            iter_count += 1
            loss_sum += loss
            
            pbar.set_description(f'loss: {loss}')
            
            if num_item_train * 0.02 <= iter_count:
                self.logger.wandbLog(
                        {'Loss' : loss_sum / float(iter_count)}
                    )
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
        
        pbar = tqdm(self.valid_loader, 'evaluation')
        with torch.no_grad():
            
            for product_id, rating, ref_ids, ref_ratings in pbar:
            
                product_id  = product_id.to(self.device)
                rating      = rating.to(self.device)
                ref_ids     = ref_ids.to(self.device)
                ref_ratings = ref_ratings.to(self.device)
                
                y = self.model(product_id, ref_ids, ref_ratings)
                y = torch.argmax(y, 1) + 1
                infer_list.append(y.cpu().squeeze())
                label_list.append(rating.cpu().squeeze())

        infer_list = torch.cat(infer_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
                    
        infer_list = all_gather(infer_list)
        label_list = all_gather(label_list)
        
        infer_list = torch.cat(infer_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        
        error = self.calculate_RMSE(infer_list, label_list)
        
        
        return error

    
    
