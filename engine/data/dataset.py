from sqlalchemy import create_engine, text

import torch
from torch.utils.data import Dataset

import numpy as np
import random

class BaseSet(Dataset):
    
    def __init__(
        self,
        table_name:str, # review_train, review_valid
        category_code:int,
        db_conn_str:str,
        max_ref_per_user:int,
        chunk_size:int
        ):
        super(BaseSet, self).__init__()

        self.table_name = table_name
        self.category_code = category_code
        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        self.total_num = None

        self.offset = 0
        self.chunk_size = chunk_size
        self.chunk = self._get_chunk(0)

    def __len__(self):
        if self.total_num is not None:
            return self.total_num
        
        with create_engine(self.db_conn_str).connect() as conn:
            res = conn.execute(
                text(
                f"""
                SELECT
                    COUNT(review_id)
                FROM
                    {self.table_name}
                WHERE
                    category_code = '{self.category_code}'
                """)
            ).fetchone()[0]

            self.total_num = res
            return res

    def _get_chunk(self, offset):

        with create_engine(self.db_conn_str).connect() as conn:
            res = conn.execute(
                text(
                f"""
                select
                    rt.target_id,
                    rt.label,
                    rtr.product_ids,
                    rtr.ratings
                from
                    (
                    select 
                        user_id,
                        product_ids,
                        ratings
                    from
                        review_train_reference rtr
                    where 
                        category_code = {self.category_code}
                    ) rtr right join (
                            SELECT 
                                user_id, 
                                product_id AS target_id, 
                                rating AS label
                            FROM 
                                {self.table_name}
                            WHERE
                                category_code = {self.category_code}
                            LIMIT {self.chunk_size} OFFSET {offset}
                        ) rt
                        on rtr.user_id = rt.user_id;  
                """
                )
            )

            return res.fetchall()
        
    def __getitem__(self, index):
        
        if not (self.offset <= index) or not (index < self.offset + len(self.chunk)):
            self.offset = index
            self.chunk = self._get_chunk(self.offset)
        
        # target_id, label, product_ids, ratings = self.chunk[index - self.offset]
        try:
            target_id, label, product_ids, ratings = self.chunk[index - self.offset]
        except Exception as e:
            print(index, self.offset, len(self.chunk))
            return None

        if product_ids is not None:
            if len(product_ids) > self.max_ref_per_user:
                product_ids = random.sample(product_ids, self.max_ref_per_user)
                ratings = random.sample(ratings, self.max_ref_per_user)

            product_ids = np.array(product_ids, dtype=np.int32)
            ratings = np.array(ratings, dtype=np.int32)

            if product_ids.shape[0] < self.max_ref_per_user:
                # padding to make all tensor sizes same
                product_ids = np.pad(product_ids, (0, self.max_ref_per_user - product_ids.shape[0]), 'constant', constant_values=(0, 0))
                ratings = np.pad(ratings, (0, self.max_ref_per_user - ratings.shape[0]), 'constant', constant_values=(0, 0))
        else:
            product_ids = np.zeros(self.max_ref_per_user, dtype=np.int32)
            ratings = np.zeros(self.max_ref_per_user, dtype=np.int32)

        return (
            torch.tensor([target_id], dtype=torch.int32), 
            torch.tensor([label], dtype=torch.int32), 
            torch.tensor(product_ids, dtype=torch.int32),
            torch.tensor(ratings, dtype=torch.int32)
        )
    
class TrainSet(BaseSet):
    
    def __init__(
        self,
        category_code:int,
        db_conn_str:str,
        max_ref_per_user:int,
        chunk_size:int
        ):
        super(TrainSet, self).__init__(
            table_name="review_train",
            category_code=category_code,
            db_conn_str=db_conn_str,
            max_ref_per_user=max_ref_per_user,
            chunk_size=chunk_size
        )
        
class ValidSet(BaseSet):
    
    def __init__(
        self,
        category_code:int,
        db_conn_str:str,
        max_ref_per_user:int,
        chunk_size:int
        ):
        super(ValidSet, self).__init__(
            table_name="review_valid",
            category_code=category_code,
            db_conn_str=db_conn_str,
            max_ref_per_user=max_ref_per_user,
            chunk_size=chunk_size
        )