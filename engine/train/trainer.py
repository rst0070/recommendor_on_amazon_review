import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from train.logger import Logger
from data.dataset import TrainSet, ValidSet
import os

class Trainer:
    
    def __init__(self, 
        category_code: int,
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
        self.category_code = category_code
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.logger = logger
        
        self.path_save_dir = path_save_dir
        
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
            rating      = rating.to(self.device).float()
            ref_ids     = ref_ids.to(self.device)
            ref_ratings = ref_ratings.to(self.device)
            
            assert len(product_id.shape)  == 2, "batch of product_id: [batch, 1]"
            assert len(rating.shape)      == 1, "batch of rating: [batch]"
            assert len(ref_ids.shape)     == 2, "batch of ref_ids: [batch, max num of reference]"
            assert len(ref_ratings.shape) == 2, "batch of ref_ratings: [batch, max num of reference]"
            
            ###################### forward and backward
            infer = self.model(product_id, ref_ids, ref_ratings)
            assert len(infer.shape)       == 1, "batch of infer: [batch]"
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
                        {
                            'category_code' : self.category_code,
                            'Loss' : loss_sum / float(iter_count)
                        }
                    )
                loss_sum = 0
                iter_count = 0
        
        ### Clear shared memory!!!!!
        TrainSet.clear_shared_memory()      
        
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

                infer_list.append(y.cpu().squeeze())
                label_list.append(rating.cpu().squeeze())
                
        ### Clear shared memory!!!
        ValidSet.clear_shared_memory()
        
        infer_list = torch.cat(infer_list, dim=0)
        label_list = torch.cat(label_list, dim=0)        
        
        error = self.calculate_RMSE(infer_list, label_list)        
        return error
    
    def calculate_RMSE(self, infers: torch.Tensor, labels: torch.Tensor) -> float:
        infers, labels = infers.float(), labels.float()
        mse = torch.mean((labels - infers) ** 2)
        return torch.sqrt(mse).numpy()
        
    
    def run(
        self,
        max_epoch:int,
        save_parameter:bool, 
        ):
        
        parameter = None
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
                        'category_code' : self.category_code,
                        'err' : err, 
                        'learning_rate': self.scheduler.get_last_lr()[0] , 
                        'epoch' : epoch
                    }
                )
                
            if err < best_err:

                best_err = err
                parameter = self.model.state_dict()
                self.logger.wandbLog({
                    'category_code' : self.category_code,
                    'err' : err,
                    'epoch' : epoch
                })
                    
        if save_parameter:
            self.save_model(
                error=best_err,
                parameter=parameter  
            )
        
                    
                
    def save_model(
        self,
        error:float,
        parameter
        ):
        if self.device != 0:
            return
        
        #directory = datetime.date.today().strftime("%Y-%m-%d")
        directory = str(self.category_code)
        directory = os.path.join(self.path_save_dir, directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        save_name = str(error)+".pt"
        full_path = os.path.join(directory, save_name)
        
        self.logger.print("saving... path: " + full_path)
        torch.save(parameter, full_path)
    


    
    
