import os
import torch
import torch.nn as nn
from train.logger import Logger
import random
import torch.utils.data as data
import numpy as np
import train.configs as configs
from data.dataset import TrainSet, ValidSet
from models.transformer import TransformerReg
from train.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def getDataLoader(dataset, batch_size, num_workers, worker_init_fn):
    return data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        pin_memory=True, shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )
    
def run(
        category_code  : int,
        wandb_key      : str,
        db_conn_str    : str
    ):
    sys_config = configs.SysConfig()
    exp_config = configs.ExpConfig()
    
    set_seed(exp_config.random_seed)
    
    # ------------------------- Logger setup
    logger = Logger(
        wandb_key       = wandb_key,
        wandb_disabled  = sys_config.wandb_disabled,
        wandb_project   = sys_config.wandb_project,
        wandb_entity    = sys_config.wandb_entity,
        wandb_name      = sys_config.wandb_name,
        wandb_notes     = sys_config.wandb_notes
    )
    
    train_dataset = TrainSet(
            category_code=category_code,
            db_conn_str = db_conn_str,
            max_ref_per_user = exp_config.max_ref_per_user_train
        )
    size = exp_config.batch_size_train // sys_config.num_workers_train
    train_dataset.preload_length(db_conn_str)
    train_loader = getDataLoader(
        dataset=train_dataset,
        batch_size=exp_config.batch_size_train, 
        num_workers=sys_config.num_workers_train,
        worker_init_fn=lambda idx: TrainSet.preload_data(
            db_conn_str=db_conn_str, 
            category_code=category_code, 
            length = TrainSet.length.value, 
            max_ref_per_user=exp_config.max_ref_per_user_train, 
            offset = idx*size, 
            size = size if idx != (sys_config.num_workers_train - 1) else size + (exp_config.batch_size_train % sys_config.num_workers_train)
            )
    )
    
    valid_dataset = ValidSet(
            category_code=category_code,
            db_conn_str = db_conn_str,
            max_ref_per_user = exp_config.max_ref_per_user_valid
        )
    size = exp_config.batch_size_valid // sys_config.num_workers_valid
    valid_dataset.preload_length(db_conn_str)
    valid_loader = getDataLoader(
        dataset=valid_dataset,
        batch_size=exp_config.batch_size_valid, 
        num_workers=sys_config.num_workers_valid,
        worker_init_fn=lambda idx: ValidSet.preload_data(
            db_conn_str=db_conn_str, 
            category_code=category_code, 
            length = TrainSet.length.value, 
            max_ref_per_user=exp_config.max_ref_per_user_valid, 
            offset = idx*size, 
            size = size if idx != (sys_config.num_workers_valid - 1) else size + (exp_config.batch_size_valid % sys_config.num_workers_valid)
            )
    )
    
    model = TransformerReg(
            num_product=sys_config.num_product,
            embedding_dim=exp_config.embedding_size,
            num_transformer_block=exp_config.num_transformer_block,
            ffn_hidden=exp_config.ffn_hidden
        ).to(sys_config.device)
    
    loss_fn = nn.MSELoss().to(sys_config.device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=exp_config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=exp_config.max_epoch,
        T_mult=1,
        eta_min=exp_config.lr_min
    )
    
    trainer = Trainer(
            category_code=category_code,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            valid_loader=valid_loader,
            logger=logger,
            path_save_dir=sys_config.path_parameter_storage,
            device=sys_config.device
        )
    
    trainer.run(
        max_epoch=exp_config.max_epoch,
        save_parameter=sys_config.save_parameter
    )

if __name__ == "__main__":
    
    # ------------------------- env setting
    from dotenv import load_dotenv
    load_dotenv()
    
    wandb_key = os.getenv('WANDB_KEY')
    assert type(wandb_key) is str
    db_conn_str = os.getenv('DB_CONN_STR')
    assert type(db_conn_str) is str
    
    for code in range(0, 33):
        print(f"Training category code: {code}")
        run(
            category_code=code,
            wandb_key=wandb_key,
            db_conn_str=db_conn_str
        )
        print(f"Category code: {code} finished!")