import wandb
import os

class Logger:
    def __init__(
            self,
            wandb_key      : str,
            wandb_disabled : bool,
            wandb_project  : str,
            wandb_entity   : str,
            wandb_name     : str,
            wandb_notes    : str,
        ):
        
        self.wandb_disabled = wandb_disabled
        
        if not self.wandb_disabled:
            os.system(f"wandb login {wandb_key}")
            wandb.init(
                project = wandb_project,
                entity  = wandb_entity,
                name    = wandb_name,
                notes   = wandb_notes
            )
        
    def wandbLog(self, contents:dict):
        if not self.wandb_disabled:
            wandb.log(contents)
    
    def print(self, *args):
        print(*args)
    
    
            