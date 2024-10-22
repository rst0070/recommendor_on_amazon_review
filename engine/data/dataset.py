from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from torch.utils.data import Dataset
from typing import Tuple

from tqdm import tqdm
import torch

import numpy as np
import pandas as pd

        

class ReferenceData:
    
    def __init__(
        self,
        db_conn_str:str,
        max_ref_per_user:int
    ):
        """
        Reference data:
            if user(the user_id) have other reviews, we can use that for explaining about the user.
            So, the reference data is other review's (product_id, rating)

        Args:
            db_conn_str (str): database connection string
            max_ref_per_user (int): maximum number of reference per user
        """
        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        
        self.cache = self._get_data()
        self.zero_ids = np.zeros(self.max_ref_per_user, dtype=np.int32)
        
    def _get_data(self):
        """
        Get reference data from reviews_reference table into DataFrame.
        """
        conn = create_engine(self.db_conn_str).connect()        
        df = pd.read_sql(
            sql="""
            SELECT user_id, product_ids, ratings
            FROM reviews_reference
            ORDER BY user_id
            """,
            con = conn.connection,
            dtype = {
                'user_id' : object # user id is int, by setting `object` pandas doesnt cast to other type
            }
        )
        df.set_index(keys='user_id', inplace=True)
        
        conn.close()    
        
        def transform_from_int(x):
            
            x = list(map(int, x))[:self.max_ref_per_user]
            x = 1 + np.array(x, dtype=np.int32)
            x = np.pad(x, (0, self.max_ref_per_user - x.shape[0]), 'constant', constant_values=(0, 0))
            
            return x
        
        def transform_from_float(x):
            
            x = list(map(int, map(float, x)))[:self.max_ref_per_user]
            x = 1 + np.array(x, dtype=np.int32)
            x = np.pad(x, (0, self.max_ref_per_user - x.shape[0]), 'constant', constant_values=(0, 0))
            
            return x
            
        
        df['product_ids'] = df['product_ids'] \
                .str.split(',') \
                .transform(
                    transform_from_int
                )
        df['ratings'] = df['ratings'] \
                .str.split(',') \
                .transform(
                    transform_from_float
                )
        
        return df
    
    def get_reference(
        self, 
        user_id, 
        except_product_id = None
        ) -> tuple[np.ndarray, np.ndarray]:
        """

        Args:
            user_id: 
                user id for other reviews
            except_product_id (_type_, optional): 
                if it is set, excepts review data of the product. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                (list of product ids, list of that's ratings), 
                the index is matched b/w ids and ratings
        """
        if user_id not in self.cache.index:
            return (
                self.zero_ids,
                self.zero_ids
            )
            
        product_ids, ratings = self.cache.loc[user_id]
            
        if except_product_id is not None:
            product_ids = product_ids.copy()
            ratings = ratings.copy()
            
            idx_delete = np.where(product_ids == except_product_id)
            product_ids[idx_delete] = 0
            ratings[idx_delete]     = 0
            
        return (
            product_ids,
            ratings
        )

class TrainSet(Dataset):
    
    def __init__(
        self,
        reference_data:ReferenceData,
        db_conn_str:str,
        max_ref_per_user:int
        ):
        super(TrainSet, self).__init__()

        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        self.reference_data = reference_data
        self.cache = self._get_cache()
    
    def _get_cache(self) -> pd.DataFrame:
        conn = create_engine(self.db_conn_str).connect()
        df = pd.read_sql(
            sql = """
            SELECT user_id, (product_id + 1) AS target_id, rating AS label
            FROM reviews_train
            """,
            con = conn.connection,
            dtype = {
                'user_id' : object,
                'target_id' : object,
                'label' : object
            }
        )
        conn.close()
        return df
        
    def __len__(self):
        return len(self.cache.index)
            
    def __getitem__(self, idx):
        user_id, target_id, label = self.cache.iloc[idx]
        
        product_ids, ratings = self.reference_data.get_reference(
                user_id=user_id,
                except_product_id=target_id
            )
            
        result = (
            torch.tensor([target_id], dtype=torch.int32), 
            torch.tensor([label], dtype=torch.float32), 
            torch.tensor(product_ids, dtype=torch.int32),
            torch.tensor(ratings, dtype=torch.int32)
        )
        return  result
            
class ValidSet(Dataset):
         
    def __init__(
        self,
        reference_data:ReferenceData,
        db_conn_str:str,
        max_ref_per_user:int
        ):
        super(ValidSet, self).__init__()

        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        self.reference_data = reference_data
        self.cache = self._get_cache()
    
    def _get_cache(self) -> pd.DataFrame:
        conn = create_engine(self.db_conn_str).connect()
        df = pd.read_sql(
            sql = """
            SELECT user_id, (product_id + 1) AS target_id, rating AS label
            FROM reviews_valid
            """,
            con = conn.connection,
            dtype = {
                'user_id' : object,
                'target_id' : object,
                'label' : object
            }
        )
        conn.close()
        return df
        
    def __len__(self):
        return len(self.cache.index)
            
    def __getitem__(self, idx):
        user_id, target_id, label = self.cache.iloc[idx]
        
        product_ids, ratings = self.reference_data.get_reference(user_id)
            
        result = (
            torch.tensor([target_id], dtype=torch.int32), 
            torch.tensor([label], dtype=torch.float32), 
            torch.tensor(product_ids, dtype=torch.int32),
            torch.tensor(ratings, dtype=torch.int32)
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
    
    reference_data = ReferenceData(sqlite_conn_str, 100)
    # train_set = TrainSet(
    #     reference_data, 
    #     sqlite_conn_str,
    #     100
    #     )
    print(reference_data.get_reference(0, 1))
    print(reference_data.get_reference(0))