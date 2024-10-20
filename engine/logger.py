import wandb
import os

class Logger:
    def __init__(
            self,
            proccess_rank  : int,
            wandb_key      : str,
            wandb_disabled : bool,
            wandb_project  : str,
            wandb_entity   : str,
            wandb_name     : str,
            wandb_notes    : str,
        ):
        """
        This logger makes sure that only proccess_rank = 0 can use wandb logging

        Args:
            proccess_rank (int): The rank
        """
        
        self.proccess_rank  = proccess_rank
        self.wandb_disabled = wandb_disabled
        
        if proccess_rank == 0 and not self.wandb_disabled:
            os.system(f"wandb login {wandb_key}")
            wandb.init(
                project = wandb_project,
                entity  = wandb_entity,
                name    = wandb_name,
                notes   = wandb_notes
            )
        
    def wandbLog(self, contents:dict):
        """
        Log to wandb

        Args:
            contents (dict): contents to log
        """
        if self.proccess_rank != 0 or self.wandb_disabled:
            return
        
        wandb.log(contents)
    
    def print(self, *args):
        if self.proccess_rank != 0:
            return
        
        print(*args)
    
    
            