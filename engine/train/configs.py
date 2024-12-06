import os

class SysConfig:
    
    def __init__(self):
        
        self.db_echo                    = False
        self.device                     = 0
        
        self.wandb_disabled             = False
        self.wandb_project              = 'Amazon_review_2023'
        self.wandb_name                 = 'transformer fixed!'
        self.wandb_entity               = 'rst0070'
        self.wandb_notes                = 'transformer small test'
        
        self.load_pretrained_parameter  = False
        self.pretrained_parameter_path  = os.path.realpath(
            os.path.join(os.path.dirname(__file__), 'parameters/best/transformer_reg3'))
        
        self.save_parameter             = True
        self.path_parameter_storage     = os.path.realpath(os.path.join(os.path.dirname(__file__), '../parameters'))
        """
        path where to save model parameter
        """
        
        self.num_workers_train          = 4
        self.num_workers_valid          = 3
        self.num_product                = 13300000 # 13.3M is maximum num in a category # home and kitchen: 3734414
        
        
class ExpConfig:
    
    def __init__(self):
        
        self.classification             = False
        
        self.max_ref_per_user_train     = 5#63#31
        self.max_ref_per_user_valid     = 31
        
        self.random_seed                = 1024
        
        
        self.embedding_size             = 8
        self.num_transformer_block      = 1
        self.ffn_hidden                 = 8
        
        
        self.batch_size_train           = 10000#7800
        self.batch_size_valid           = 3000
        self.max_epoch                  = 3
        
        self.lr                         = 8 * 1e-4 #8 * 1e-4
        self.lr_min                     = 1e-6 
        # self.lr_step                    = 0.98
        
        