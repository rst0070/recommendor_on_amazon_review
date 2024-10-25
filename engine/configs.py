import os

class SysConfig:
    
    def __init__(self):
        
        self.db_echo                    = False
        
        self.wandb_disabled             = False
        self.wandb_project              = 'Amazon_review_2023'
        self.wandb_name                 = 'NCF with transformer'
        self.wandb_entity               = 'rst0070'
        self.wandb_notes                = 'transformer_reg6, the best model setting'
        
        self.load_pretrained_parameter  = False
        self.pretrained_parameter_path  = os.path.realpath(
            os.path.join(os.path.dirname(__file__), 'parameters/best/1.0740489959716797.pt'))
        
        self.save_parameter             = True
        self.path_parameter_storage     = os.path.realpath(os.path.join(os.path.dirname(__file__), 'parameters/best/transformer_reg6'))
        """
        path where to save model parameter
        """
        
        self.num_workers_train          = 2
        self.num_workers_valid          = 2
        self.num_product                = 3734414
        self.num_rating                 = 5
        
        
class ExpConfig:
    
    def __init__(self):
        
        self.classification             = False
        
        self.max_ref_per_user_train     = 63#63#31
        self.max_ref_per_user_valid     = 100
        
        self.random_seed                = 1024
        
        
        self.embedding_size             = 32
        self.num_transformer_block      = 5
        self.ffn_hidden                 = 80
        
        
        self.batch_size_train           = 7800#7800
        self.batch_size_valid           = 1000
        self.max_epoch                  = 100
        
        self.lr                         = 8 * 1e-4
        self.lr_min                     = 1e-6 
        
        