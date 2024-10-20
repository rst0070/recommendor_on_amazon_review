import os

class SysConfig:
    
    def __init__(self):
        
        self.db_echo                    = False
        
        self.wandb_disabled             = False
        self.wandb_project              = 'Amazon_review_2023'
        self.wandb_name                 = 'NCF with transformer'
        self.wandb_entity               = 'rst0070'
        self.wandb_notes                = 'lr=8*1e-4, max_ref_per_user=63, emb_size = 32, batch_size = 100'
        
        self.path_parameter_storage     = os.path.realpath(os.path.join(os.path.dirname(__file__), 'parameters'))
        """
        path where to save model parameter
        """
        self.num_workers                = 0
        
        self.num_product                = 3734413 + 1 + 1 
        """
        +1 to maximum id of products (because it starts from 0)
        
        +1 for padding embedding        
        """
        self.num_rating                 = 5 + 1
        """
        +1 is for padding embedding
        """
        
        
class ExpConfig:
    
    def __init__(self):
        
        self.max_ref_per_user           = 63
        
        self.random_seed                = 1024
        
        
        self.embedding_size             = 32
        self.num_transformer_block      = 5
        self.ffn_hidden                 = 80
        
        
        self.batch_size_train           = 100
        self.batch_size_valid           = 100
        self.max_epoch                  = 100
        
        self.lr                         = 8 * 1e-4
        self.lr_min                     = 1e-6 # this could not work because i turned off scheduler in some cases
        
        