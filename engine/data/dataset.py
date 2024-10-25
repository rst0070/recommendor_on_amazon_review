from sqlalchemy import create_engine

from torch.utils.data import Dataset
import torch

import numpy as np
import pandas as pd

        

class ReferenceData:
    
    def __init__(
        self,
        db_conn_str:str
    ):
        """
        Reference data:
            if user(the user_id) have other reviews, we can use that for explaining about the user.
            So, the reference data is other review's (product_id, rating)

        Args:
            db_conn_str (str): database connection string
        """
        self.db_conn_str = db_conn_str
        
        self.cache = self._get_data()
        
    def _get_data(self):
        """
        Get reference data from reviews_reference table into DataFrame.
        """
        conn = create_engine(self.db_conn_str).connect()        
        df = pd.read_sql(
            sql="""
            SELECT user_id, review_ids, product_ids, ratings
            FROM reviews_reference
            ORDER BY user_id
            """,
            con = conn.connection,
            dtype = {
                'user_id' : object, # user id is int, by setting `object` pandas doesnt cast to other type
                'review_ids' : object,
                'product_ids' : object,
                'ratings': object
            }
        )
        conn.close()    
        df.set_index(keys='user_id', inplace=True)
        return df
    
    def _transform_from_review_id_str(self, x, num_max_ref) -> np.ndarray:
        x = eval(x)
        
        if type(x) is not tuple:
            x = (x,)
            
        x = list(map(int, x))[:num_max_ref]
            
        return np.array(x, dtype=np.int32)
    
    def _transform_from_product_id_str(self, x, num_max_ref) -> np.ndarray:
        x = eval(x)
        
        if type(x) is not tuple:
            x = (x,)
            
        x = list(map(int, x))[:num_max_ref]
        
        x = 1 + np.array(x, dtype=np.int32) # id = 0 is for padding
        x = np.pad(x, (0, num_max_ref - x.shape[0]), 'constant', constant_values=(0, 0))
            
        return x
        
    def _transform_from_rating_str(self,x, num_max_ref) -> np.ndarray:
        x = eval(x)
            
        if type(x) is not tuple:
            x = (x,)
            
        x = list(map(int, map(float, x)))[:num_max_ref]
            
        x = np.array(x, dtype=np.int32) # rating: [1,5]
        x = np.pad(x, (0, num_max_ref - x.shape[0]), 'constant', constant_values=(0, 0))
            
        return x
        
    def get_reference(
        self, 
        user_id, 
        num_max_ref: int,
        except_review_id = None
        ) -> tuple[np.ndarray, np.ndarray]:
        """

        Args:
            user_id: 
                user id for other reviews
            except_review_id (_type_, optional): 
                if it is set,this function excepts review data of the review id. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: 
                (list of product ids, list of that's ratings), 
                the index is matched b/w ids and ratings
        """
        if user_id not in self.cache.index:
            return (
                np.zeros(num_max_ref, dtype=np.int32),
                np.zeros(num_max_ref, dtype=np.int32)
            )
            
        review_ids, product_ids, ratings = self.cache.loc[user_id]
                
        product_ids = self._transform_from_product_id_str(product_ids, num_max_ref)
        ratings     = self._transform_from_rating_str(ratings, num_max_ref)
            
        if except_review_id is not None:
            # delete information of review which id is `except_review_id`
            #
            
            review_ids  = self._transform_from_review_id_str(review_ids, num_max_ref) # only used in this function
            
            idx_delete = np.where(review_ids == except_review_id)
            
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
            SELECT user_id, review_id, (product_id + 1) AS target_id, rating AS label
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
        user_id, review_id, target_id, label = self.cache.iloc[idx]
        
        product_ids, ratings = self.reference_data.get_reference(
                user_id=user_id,
                num_max_ref=self.max_ref_per_user,
                except_review_id=review_id
            )
            
        result = (
            torch.tensor([target_id], dtype=torch.int32), 
            torch.tensor([label], dtype=torch.int32), 
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
        
        product_ids, ratings = self.reference_data.get_reference(user_id, self.max_ref_per_user)
            
        result = (
            torch.tensor([target_id], dtype=torch.int32), 
            torch.tensor([label], dtype=torch.int32), 
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
    
    reference_data = ReferenceData(sqlite_conn_str)
    # train_set = TrainSet(
    #     reference_data, 
    #     sqlite_conn_str,
    #     100
    #     )
    print(reference_data.get_reference(0, 1))
    print(reference_data.get_reference(0, 1000))