from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import os
import random
from typing import Tuple
from tqdm import tqdm


USER_ID = 0
PRODUCT_ID = 0
dict_user_id = {}
dict_product_id = {}
dict_train_ref = {}
"""
key: user_id
value: {'review_ids':'1,2,3','product_ids':'5,6,7', 'ratings':'1.0,3.0,5.0'}
"""

def get_ids(user_id:str, product_id:str) -> Tuple[int, int]:
    """
    Generates integer ids for user_id and product_id
    from string ids
    
    Args:
        user_id (str): _description_
        product_id (str): _description_

    Returns:
        Tuple[int, int]: user_id, product_id
    """
    global USER_ID
    global PRODUCT_ID
    global dict_user_id
    global dict_product_id
    
    if user_id not in dict_user_id.keys():
            
        dict_user_id[user_id] = USER_ID
        USER_ID += 1
            
    if product_id not in dict_product_id.keys():
            
        dict_product_id[product_id] = PRODUCT_ID
        PRODUCT_ID += 1
        
    return dict_user_id[user_id], dict_product_id[product_id]
        
def _insert_reference_data_batch(engine:Engine, batch_data:list):
    
    query = text(
        """
        INSERT INTO reviews_reference(user_id, review_ids, product_ids, ratings)
        VALUES (:user_id, :review_ids, :product_ids, :ratings)
        """
    )
    
    with engine.begin() as conn:
        conn.execute(
            query,
            batch_data
        )    
    
    
def insert_reference_data(engine:Engine, batch_size:int):
    """
    Updates reference data.
    It means inserting all review data per a user, 
    making it possible to use products as user's feature.
    """
    batch = []
    for user_id, val_dict in tqdm(dict_train_ref.items(), desc="update ref data"):
        
        batch.append(
            {
                'user_id'       : user_id,
                'review_ids'    : val_dict['review_ids'],
                'product_ids'   : val_dict['product_ids'],
                'ratings'       : val_dict['ratings']
            }
        )
        
        if len(batch) == batch_size:
            _insert_reference_data_batch(engine, batch)
            batch.clear()
            
    if batch:
        _insert_reference_data_batch(engine, batch)
        batch.clear()

        

def _insert_reviews_batch(engine:Engine, batch_data:list):
    """
    sub function for insert_reviews.
    - batch insert
    - if the batch goes to reviews_train
        - save the review data for reference data
    """
    target_table = random.choices(
                        ["reviews_train","reviews_test","reviews_valid"],
                        [8.0,1.0,1.0],
                        k = 1
                    )[0]
        
    with engine.begin() as conn:
        conn.execute(
            text(
            f"""
            INSERT INTO {target_table}(review_id,user_id,product_id,rating)
            VALUES(:review_id,:user_id,:product_id,:rating)
            """),
            batch_data
        )
            
    if target_table != "reviews_train":
        return
    
    for row_dict in batch_data:
        
        prefix = ','
        user_id = row_dict['user_id']    
        if user_id not in dict_train_ref.keys():
            dict_train_ref[user_id] = {'review_ids':'','product_ids':'', 'ratings':''}
            prefix = ''
        
        dict_train_ref[user_id]['review_ids']   += prefix + str(row_dict['review_id'])
        dict_train_ref[user_id]['product_ids']  += prefix + str(row_dict['product_id'])
        dict_train_ref[user_id]['ratings']      += prefix + str(row_dict['rating'])

def insert_reviews(
        review_dataset,
        engine:Engine,
        batch_size:int
    ):
    """
    Insert reviews to reviews_train, reviews_test, reviews_valid

    Args:
        review_dataset (_type_): iteratable
        engine (Engine): db engine
        batch_size (int): 
    """
    
    assert 'user_id' in review_dataset[0].keys() # string type
    assert 'parent_asin' in review_dataset[0].keys() # string type
    assert 'rating' in review_dataset[0].keys()
    
    batch = []
    
    review_id = 0
    for review in tqdm(review_dataset, desc="inserting review into db"):
        
        user_id, product_id, rating = review['user_id'], review['parent_asin'], float(review['rating'])
        user_id, product_id = get_ids(user_id=user_id, product_id=product_id)
        
        assert type(user_id) is int
        assert type(product_id) is int
        assert type(rating) is float        
        
        # --------------------- save to batch....
        row = {
            'review_id' : review_id,
            'user_id'   : user_id,
            'product_id': product_id,
            'rating'    : rating
        }
        
        batch.append(row)
        
        if len(batch) == batch_size:
            _insert_reviews_batch(engine=engine, batch_data=batch)
            batch.clear()
            
        review_id += 1
        
    if batch:
        _insert_reviews_batch(engine=engine, batch_data=batch)
        batch.clear()
        
def create_schema(engine:Engine):
    with engine.begin() as conn:
        
        for name in ['reviews_train','reviews_test','reviews_valid']:
            conn.execute(
                text(
                f"""
                CREATE TABLE {name} (
                    review_id   integer primary key,
                    user_id     integer,
                    product_id  integer,
                    rating      real
                )
                """
                )
            )
            
        conn.execute(
            text(
            """
            CREATE TABLE reviews_reference(
                user_id     integer primary key,
                review_ids  text,
                product_ids text,
                ratings     text
            )
            """
            )
        )

if __name__ == "__main__":
        
    print("Starting preprocess...")
    db_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "data/database.db"))
    sqlite_conn_str = f"sqlite:////{db_path}"
    engine = create_engine(sqlite_conn_str)
    assert type(engine) is Engine
    
    create_schema(engine)
    print("create schema: done!")
    # --------------------- load review dataset
    # ---------------------
    cache_dir_path = os.path.join(os.path.dirname(__file__), 'cache')
    
    from datasets import load_dataset
    
    review_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_review_Home_and_Kitchen",
                        trust_remote_code=True,
                        cache_dir=cache_dir_path,
                        streaming=False
                    )['full']
    print("load review dataset: done!")
    
    # --------------------- insert review data
    # ---------------------
    insert_reviews(review_dataset, engine, 10000)
    print("insert review data: done!")
    
    # --------------------- insert reference data
    # ---------------------
    insert_reference_data(engine, 100)
    
    
        
    