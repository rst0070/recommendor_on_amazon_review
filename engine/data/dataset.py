from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from torch.utils.data import Dataset
from typing import Tuple

from tqdm import tqdm
import torch
import time

class CacheStorage:
    
    cach_ref = None
    
    def __init__(
        self,
        db_conn_str:str,
        max_ref_per_user:int
        ):
        
        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        self.cache_reviews_train: list[tuple] = []
        self.cache_reviews_valid: list[tuple] = []
        self.cache_ref: dict[int,dict] = {}
        
        self._prepare_cache_review()
        self._prepare_cache_ref()
    
    def _prepare_cache_review(self):
        
        with create_engine(self.db_conn_str).connect() as conn:
            self.cache_reviews_train = conn.execute(
                text(
                f"""
                SELECT user_id, product_id + 1, rating + 1
                FROM reviews_train
                """
                )
            ).all()
        
        print("caching train done!")

        with create_engine(self.db_conn_str).connect() as conn:
            self.cache_reviews_valid = conn.execute(
                text(
                f"""
                SELECT user_id, product_id + 1, rating + 1
                FROM reviews_valid
                """
                )
            ).all()
            
        print("caching valid done!")
        
        
    def _prepare_cache_ref(self):
        res = []
        with create_engine(self.db_conn_str).connect() as conn:
            
                
            res = conn.execute(text(f"""
                SELECT user_id, product_ids, ratings
                FROM reviews_reference
                """)).all()
            
            
        for user_id, p_s, r_s in tqdm(res):
            p_s = p_s.split(',')[0:self.max_ref_per_user]
            r_s = r_s.split(',')[0:self.max_ref_per_user]
            
            if len(p_s) < self.max_ref_per_user: # p_s and r_s have same length
                for i in range(len(p_s), self.max_ref_per_user):
                    p_s.append(0)
                    r_s.append(0)
                    
            
            self.cache_ref[user_id] = {
                    'product_ids':[ int(p_id) + 1 for p_id in p_s],
                    'ratings':[ int(float(r)) + 1 for r in r_s]
                }
    
        

class TrainSet(Dataset):
    
    def __init__(
        self,
        cache:CacheStorage,
        max_ref_per_user:int
        ):
        super(TrainSet, self).__init__()

        self.cache_review: list = cache.cache_reviews_train
        self.cache_ref: dict[int,dict] = cache.cache_ref
        
        self.max_ref_per_user = max_ref_per_user
    
            
    def __len__(self):
        return len(self.cache_review)
            
    def __getitem__(self, idx):
        user_id, product_id, rating = self.cache_review[idx]
        
        
        if user_id not in self.cache_ref.keys():
            
            self.cache_ref[user_id] = {
                'product_ids': [0 for i in range(0, self.max_ref_per_user)],
                'ratings' : [0 for i in range(0, self.max_ref_per_user)]
            }
            
        
        product_ids = self.cache_ref[user_id]['product_ids']
        ratings = self.cache_ref[user_id]['ratings']
            
        result = (
            torch.tensor([product_id], dtype=torch.int), 
            torch.tensor([rating], dtype=torch.float32), 
            torch.tensor(product_ids, dtype=torch.int),
            torch.tensor(ratings, dtype=torch.int)
        )
        return  result
            
class ValidSet(Dataset):
         
    def __init__(
        self,
        cache:CacheStorage,
        max_ref_per_user:int
        ):
        super(ValidSet, self).__init__()

        self.cache_review: list = cache.cache_reviews_valid
        self.cache_ref: dict[int,dict] = cache.cache_ref

        self.max_ref_per_user = max_ref_per_user
            
    def __len__(self):
        return len(self.cache_review)
            
    def __getitem__(self, idx):
        user_id, product_id, rating = self.cache_review[idx]
        
        
        if user_id not in self.cache_ref.keys():
            
            self.cache_ref[user_id] = {
                'product_ids': [0 for i in range(0, self.max_ref_per_user)],
                'ratings' : [0 for i in range(0, self.max_ref_per_user)]
            }
            
        
        product_ids = self.cache_ref[user_id]['product_ids']
        ratings = self.cache_ref[user_id]['ratings']
            
        result = (
            torch.tensor([product_id], dtype=torch.int), 
            torch.tensor([rating], dtype=torch.float32), 
            torch.tensor(product_ids, dtype=torch.int),
            torch.tensor(ratings, dtype=torch.int)
        )
        return  result
                    
        
            
    
if __name__ == "__main__":
    """
    Simple test for dataset    
    """
    from sqlalchemy import create_engine
    from os import path
    
    db_path = path.realpath(path.join(path.dirname(__file__), "database.db"))
    sqlite_conn_str = f"sqlite:////{db_path}"
    