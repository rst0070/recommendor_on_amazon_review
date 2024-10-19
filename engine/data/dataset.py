from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from torch.utils.data import Dataset
from typing import Tuple

from tqdm import tqdm
import torch
import time

class TrainSet(Dataset):
    
    def __init__(
        self,
        db_engine:Engine,
        max_ref_per_user:int
        ):
        super(TrainSet, self).__init__()

        self.db_engine = db_engine
        self.max_ref_per_user = max_ref_per_user
        
        self.cache_review: list[tuple] = []
        self.cache_ref: dict[int,dict] = {}
        
        start_time = time.time()
        print("selecting reviews...")
        with db_engine.connect() as conn:
            res = conn.execute(
                text(
                """
                SELECT user_id, product_id, rating
                FROM reviews_train
                """
                )
            )
            
            self.cache_review = res.all()
            
        print(f"selecting secs: {time.time() - start_time}")
        
        
    def __len__(self):
        return len(self.cache_review)
            
    def __getitem__(self, idx):
        user_id, product_id, rating = self.cache_review[idx]
        
        if user_id not in self.cache_ref.keys():
            self.cache_ref[user_id] = {'product_ids':[], 'ratings':[]}
            
            p_s, r_s = '', ''
            with self.db_engine.connect() as conn:
                p_s, r_s = conn.execute(text(f"""
                                SELECT product_ids, ratings FROM reviews_reference WHERE user_id = {user_id}
                            """)).fetchone()
            
            if len(p_s) > 0:
                p_s = p_s.split(',')[0:self.max_ref_per_user]
                self.cache_ref[user_id]['product_ids'] = [ int(p_id) for p_id in p_s]
                
            if len(r_s) > 0:
                r_s = r_s.split(',')[0:self.max_ref_per_user]
                self.cache_ref[user_id]['ratings'] = [ float(r) for r in r_s]
        
        result = (
            product_id, 
            rating, 
            self.cache_ref[user_id]['product_ids'],
            self.cache_ref[user_id]['ratings']
        )
        return  result
            
            
                
                    
        
            
    
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from os import path
    
    db_path = path.realpath(path.join(path.dirname(__file__), "database.db"))
    sqlite_conn_str = f"sqlite:////{db_path}"
    
    engine = create_engine(sqlite_conn_str)
    train_set = TrainSet(engine, 100)
    
    print(len(train_set))
    
    ptime = time.time()
    print(train_set[90])
    print(time.time() - ptime)