import torch.multiprocessing
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import torch.nn as nn
from trainer import Trainer
from models.ncf import Ncf
from logger import Logger
import wandb
import random
import torch.utils.data as data
import datetime
import numpy as np
import configs
from data.dataset import CacheStorage, TrainSet, ValidSet
from sqlalchemy.engine import Engine, Connection

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def getDataLoader(dataset, batch_size, num_workers):
    return data.DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           pin_memory=True, shuffle=False,
                           sampler=DistributedSampler(dataset),
                           num_workers=num_workers
                           )

def run(
        rank           : int,
        world_size     : int, 
        port           : int,
        wandb_key      : str,
        db_conn_str    : str
    ):
    
    # ------------------------- DDP setup
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device: int = rank
    
    # ------------------------- Get configs & seed  setup
    sys_config = configs.SysConfig()
    exp_config = configs.ExpConfig()
    
    set_seed(exp_config.random_seed)
    
    # ------------------------- Logger setup
    logger = Logger(
        proccess_rank   = device,
        wandb_key       = wandb_key,
        wandb_disabled  = sys_config.wandb_disabled,
        wandb_project   = sys_config.wandb_project,
        wandb_entity    = sys_config.wandb_entity,
        wandb_name      = sys_config.wandb_name,
        wandb_notes     = sys_config.wandb_notes
    )
        
    # ------------------------- Data sets
    cache_storage = CacheStorage(
            db_conn_str = db_conn_str,
            max_ref_per_user = exp_config.max_ref_per_user
        )
    
    train_loader = getDataLoader(
            dataset=TrainSet(
                    cache=cache_storage,
                    max_ref_per_user = exp_config.max_ref_per_user    
                ),
            batch_size=exp_config.batch_size_train, 
            num_workers=sys_config.num_workers
        )
    
    test_loader = getDataLoader(
            dataset=ValidSet(
                    cache=cache_storage,
                    max_ref_per_user = exp_config.max_ref_per_user  
                ), 
            batch_size=exp_config.batch_size_valid, 
            num_workers=sys_config.num_workers
        )
    
    # ------------------------- Model setup
    model = DDP(
        Ncf(
            num_product=sys_config.num_product,
            num_rating=sys_config.num_rating,
            embedding_dim=exp_config.embedding_size,
            num_transformer_block=exp_config.num_transformer_block,
            ffn_hidden=exp_config.ffn_hidden,
            device = device
            ).to(device) )
    
    loss_fn = nn.MSELoss().to(device) #DDP is not needed when a module doesn't have any parameter that requires a gradient.
    
    # ------------------------- optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=exp_config.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=exp_config.max_epoch,
    #     T_mult=1,
    #     eta_min=exp_config.lr_min
    # )
    
    
    # ------------------------- trainer
    trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            logger=logger,
            device=device
        )
    
    
    trainer.run(
        max_epoch=exp_config.max_epoch,
        path_parameter_storage=sys_config.path_parameter_storage
    )
    
    destroy_process_group()

if __name__ == "__main__":
    
    # ------------------------- env setting
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    load_dotenv()
    
    wandb_key = os.getenv('WANDB_KEY')
    _db_path = os.path.realpath(os.path.join(
        os.path.dirname(__file__),
        'data/database.db'
    ))
    db_conn_str = f"sqlite:////{_db_path}"
    
    
    # ------------------------- torch setting
    sys_config, exp_config = configs.SysConfig(), configs.ExpConfig()
    
    set_seed(exp_config.random_seed)
    torch.cuda.empty_cache()
    
    port = f'10{datetime.datetime.now().microsecond % 100}'
    world_size = torch.cuda.device_count()
    
    
    # ------------------------- run
    torch.multiprocessing.spawn(
            run, 
            args = (
                world_size, 
                port,
                wandb_key,
                db_conn_str
            ),
            nprocs=world_size
        )