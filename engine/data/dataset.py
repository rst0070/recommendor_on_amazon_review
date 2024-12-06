from sqlalchemy import create_engine, text
import ctypes
import torch
from torch.utils.data import Dataset
import multiprocessing as mp
import numpy as np
import random

class TrainSet(Dataset):
    
    length           = None
    target_ids       = None
    labels           = None
    product_ids_list = None
    ratings_list     = None
    
    def __init__(
        self,
        category_code:int,
        db_conn_str:str,
        max_ref_per_user:int
        ):
        super(TrainSet, self).__init__()
        
        self.category_code = category_code
        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        
    def __len__(self):
        return TrainSet.length.value
    
    def __getitem__(self, idx):
        return (
            TrainSet.target_ids[idx],
            TrainSet.labels[idx],
            TrainSet.product_ids_list[idx],
            TrainSet.ratings_list[idx]
        )
    
    def preload_length(self, db_conn_str):
        with create_engine(db_conn_str).connect() as conn:
            length = conn.execute(
                text(
                f"""
                SELECT
                    COUNT(review_id)
                FROM
                    review_train
                WHERE
                    category_code = '{self.category_code}'
                """)
            ).fetchone()[0]

            TrainSet.length = mp.Value('i', length)
    
    @classmethod
    def clear_shared_memory(cls):
        try:
            cls.tmp_target_ids.close()
            cls.tmp_target_ids.unlink()
            cls.tmp_labels.close()
            cls.tmp_labels.unlink()
            cls.tmp_product_ids_list.close()
            cls.tmp_product_ids_list.unlink() 
            cls.tmp_ratings_list.close()
            cls.tmp_ratings_list.unlink()
        except:
            pass
        
    @classmethod
    def preload_data(
        cls, 
        db_conn_str,
        category_code,
        length, 
        max_ref_per_user,
        offset,
        size
        ):
        # 1. release existing shared memory
        cls.clear_shared_memory()
        # 2. set shared memory
        cls.tmp_target_ids       = mp.Array(ctypes.c_int32, length)
        cls.tmp_labels           = mp.Array(ctypes.c_int32, length)
        cls.tmp_product_ids_list = mp.Array(ctypes.c_int32, length * max_ref_per_user)
        cls.tmp_ratings_list     = mp.Array(ctypes.c_int32, length * max_ref_per_user)
        
        cls.target_ids       = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_target_ids.get_obj()).reshape(length, 1) )
        cls.labels           = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_labels.get_obj()).reshape(length) )
        cls.product_ids_list = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_product_ids_list.get_obj()).reshape(length, max_ref_per_user) )
        cls.ratings_list     = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_ratings_list.get_obj()).reshape(length, max_ref_per_user) )
        
        # 3. get all data
        with create_engine(db_conn_str).connect() as conn:
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
                        category_code = {category_code}
                    ) rtr right join (
                            SELECT 
                                user_id, 
                                product_id AS target_id, 
                                rating AS label
                            FROM 
                                review_train
                            WHERE
                                category_code = {category_code}
                            LIMIT {size} OFFSET {offset}
                        ) rt
                        on rtr.user_id = rt.user_id;  
                """
                )
            )
            
            for idx, (target_id, label, product_ids, ratings) in enumerate(res):
                if product_ids is not None:
                    if len(product_ids) > max_ref_per_user:
                        product_ids = random.sample(product_ids, max_ref_per_user)
                        ratings = random.sample(ratings, max_ref_per_user)

                    product_ids = np.array(product_ids, dtype=np.int32)
                    ratings = np.array(ratings, dtype=np.int32)

                    if product_ids.shape[0] < max_ref_per_user:
                        # padding to make all tensor sizes same
                        product_ids = np.pad(product_ids, (0, max_ref_per_user - product_ids.shape[0]), 'constant', constant_values=(0, 0))
                        ratings = np.pad(ratings, (0, max_ref_per_user - ratings.shape[0]), 'constant', constant_values=(0, 0))
                else:
                    product_ids = np.zeros(max_ref_per_user, dtype=np.int32)
                    ratings = np.zeros(max_ref_per_user, dtype=np.int32)
                
                cls.target_ids[idx, 0]    = torch.tensor(target_id, dtype=torch.int32)
                cls.labels[idx]           = torch.tensor(label, dtype=torch.int32)
                cls.product_ids_list[idx] = torch.tensor(product_ids, dtype=torch.int32)
                cls.ratings_list[idx]     = torch.tensor(ratings, dtype=torch.int32)
                
class ValidSet(Dataset):
    
    length           = None
    target_ids       = None
    labels           = None
    product_ids_list = None
    ratings_list     = None
    
    def __init__(
        self,
        category_code:int,
        db_conn_str:str,
        max_ref_per_user:int
        ):
        super(ValidSet, self).__init__()
        
        self.category_code = category_code
        self.db_conn_str = db_conn_str
        self.max_ref_per_user = max_ref_per_user
        
    def __len__(self):
        return ValidSet.length.value
    
    def __getitem__(self, idx):
        return (
            ValidSet.target_ids[idx],
            ValidSet.labels[idx],
            ValidSet.product_ids_list[idx],
            ValidSet.ratings_list[idx]
        )
        
    def preload_length(self, db_conn_str):
        with create_engine(db_conn_str).connect() as conn:
            length = conn.execute(
                text(
                f"""
                SELECT
                    COUNT(review_id)
                FROM
                    review_valid
                WHERE
                    category_code = '{self.category_code}'
                """)
            ).fetchone()[0]

            ValidSet.length = mp.Value('i', length)
            
    @classmethod
    def clear_shared_memory(cls):
        try:
            cls.tmp_target_ids.close()
            cls.tmp_target_ids.unlink()
            cls.tmp_labels.close()
            cls.tmp_labels.unlink()
            cls.tmp_product_ids_list.close()
            cls.tmp_product_ids_list.unlink() 
            cls.tmp_ratings_list.close()
            cls.tmp_ratings_list.unlink()
        except:
            pass      
        
    @classmethod
    def preload_data(
        cls, 
        db_conn_str,
        category_code,
        length, 
        max_ref_per_user,
        offset,
        size
        ):
        # 1. release existing shared memory
        cls.clear_shared_memory()
        # 2. set shared memory
        cls.tmp_target_ids       = mp.Array(ctypes.c_int32, length)
        cls.tmp_labels           = mp.Array(ctypes.c_int32, length)
        cls.tmp_product_ids_list = mp.Array(ctypes.c_int32, length * max_ref_per_user)
        cls.tmp_ratings_list     = mp.Array(ctypes.c_int32, length * max_ref_per_user)
        
        cls.target_ids       = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_target_ids.get_obj()).reshape(length, 1) )
        cls.labels           = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_labels.get_obj()).reshape(length) )
        cls.product_ids_list = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_product_ids_list.get_obj()).reshape(length, max_ref_per_user) )
        cls.ratings_list     = torch.from_numpy( np.ctypeslib.as_array(cls.tmp_ratings_list.get_obj()).reshape(length, max_ref_per_user) )
        
        # 3. get all data
        with create_engine(db_conn_str).connect() as conn:
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
                        category_code = {category_code}
                    ) rtr right join (
                            SELECT 
                                user_id, 
                                product_id AS target_id, 
                                rating AS label
                            FROM 
                                review_valid
                            WHERE
                                category_code = {category_code}
                            LIMIT {size} OFFSET {offset}
                        ) rt
                        on rtr.user_id = rt.user_id;  
                """
                )
            )
            
            for idx, (target_id, label, product_ids, ratings) in enumerate(res):
                if product_ids is not None:
                    if len(product_ids) > max_ref_per_user:
                        product_ids = random.sample(product_ids, max_ref_per_user)
                        ratings = random.sample(ratings, max_ref_per_user)

                    product_ids = np.array(product_ids, dtype=np.int32)
                    ratings = np.array(ratings, dtype=np.int32)

                    if product_ids.shape[0] < max_ref_per_user:
                        # padding to make all tensor sizes same
                        product_ids = np.pad(product_ids, (0, max_ref_per_user - product_ids.shape[0]), 'constant', constant_values=(0, 0))
                        ratings = np.pad(ratings, (0, max_ref_per_user - ratings.shape[0]), 'constant', constant_values=(0, 0))
                else:
                    product_ids = np.zeros(max_ref_per_user, dtype=np.int32)
                    ratings = np.zeros(max_ref_per_user, dtype=np.int32)
                
                cls.target_ids[idx, 0]    = torch.tensor(target_id, dtype=torch.int32)
                cls.labels[idx]           = torch.tensor(label, dtype=torch.int32)
                cls.product_ids_list[idx] = torch.tensor(product_ids, dtype=torch.int32)
                cls.ratings_list[idx]     = torch.tensor(ratings, dtype=torch.int32)