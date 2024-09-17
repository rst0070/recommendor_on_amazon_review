from typing import Tuple, Dict
from pyspark import SparkContext, RDD

def load_datasets():
    #!/usr/bin/env python
    import os

    # Set cache directory under the current directory
    cwd = os.getcwd()
    cache_dir = os.path.join(cwd, "cache")
    os.environ["HF_HOME"] = cache_dir
    # Load the dataset
    from datasets import load_dataset

    review_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_review_Home_and_Kitchen",
                        trust_remote_code=True,
                        streaming=True
                    )['full']

    item_meta_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_meta_Home_and_Kitchen",
                        split="full",
                        trust_remote_code=True,
                        streaming=True
                    )
    
    """
        <User's review data>
        {
            'rating':5.0,
            'title':'Such a lovely scent but not overpowering.',
            'text':"This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!",
            'images':[],
            'asin':'B00YQ6X8EO',
            'parent_asin':'B00YQ6X8EO',
            'user_id':'AGKHLEW2SOWHNMFQIJGBECAF7INQ',
            'timestamp':1588687728923,
            'helpful_vote':0,
            'verified_purchase':True
        }


        <Item meta data>
        {
            'main_category':'All Beauty',
            'title':'Howard LC0008 Leather Conditioner,8-Ounce (4-Pack)',
            'average_rating':4.8,
            'rating_number':10,
            'features':[],
            'description':[],
            'price':'None',
            'images':
                {
                    'hi_res':[None,'https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg'],
                    'large':['https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg','https://m.media-amazon.com/images/I/41w2yznfuZL.jpg'],
                    'thumb':['https://m.media-amazon.com/images/I/41qfjSfqNyL._SS40_.jpg','https://m.media-amazon.com/images/I/41w2yznfuZL._SS40_.jpg'],
                    'variant':['MAIN','PT01']
                },
            'videos':{'title':[],'url':[],'user_id':[]},
            'store':'Howard Products',
            'categories':[],
            'details':'{"Package Dimensions":"7.1 x 5.5 x 3 inches; 2.38 Pounds","UPC":"617390882781"}',
            'parent_asin':'B01CUPMQZE',
            'bought_together':None,
            'subtitle':None,
            'author':None
        }
    """
    
    return review_dataset, item_meta_dataset

def load_table(sc:SparkContext) -> Tuple[RDD, RDD]:
    """
    
    returns train_rdd, test_rdd. 
    RDDs are in form of (user_id, item_id, rating).

    
    Args:
        sc (SparkContext): _description_
    """
    
    review_dataset, _ = load_datasets()
    
    review_rdd = sc.parallelize(review_dataset)
    
    def dict_to_tuple(review: Dict):
        return review['user_id'], review['parent_asin'], review['rating']
    
    review_rdd = review_rdd.map(dict_to_tuple)
    
    seed = 1234
    train_fraction = 0.9  # 90% for training, 10% for testing
    train_rdd, test_rdd = review_rdd.randomSplit([train_fraction, 1 - train_fraction], seed=seed)
    
    return train_rdd, test_rdd
    



# Define a function to process the data and save to a database
def save_to_database(batch_size=1000):
    
    import sqlite3
    import random
    from tqdm import tqdm
    
    con = sqlite3.connect("reviews.db")
    cur = con.cursor()
    
    cur.execute("""
            CREATE TABLE reviews_train(
                row_id integer primary key,
                user_id TEXT,
                product_id TEXT,
                timestamp TEXT,
                rating REAL
            );            
            """)
    
    cur.execute("""
            CREATE TABLE reviews_test(
                row_id integer primary key,
                user_id TEXT,
                product_id TEXT,
                timestamp TEXT,
                rating REAL
            );            
            """)
    
    dataset, _ = load_datasets()
    
    # Initialize batch processing
    batch = []
    
    # Process the dataset in batches
    for review in tqdm(dataset):
        # Assuming dataset has keys like 'review_id', 'review_text', etc.
        batch.append(
            (
                review['user_id'],
                review['parent_asin'],
                str(review['timestamp']),
                review['rating']
            )
        )
        
        # Once the batch size is reached, insert into the database
        if len(batch) == batch_size:
            t_or_t = random.randint(1, 10)
            
            if t_or_t == 1:
                cur.executemany("insert into reviews_test(row_id, user_id, product_id, timestamp, rating) values (NULL,?,?,?,?)", batch)   
            else:
                cur.executemany("insert into reviews_train(row_id, user_id, product_id, timestamp, rating) values (NULL,?,?,?,?)", batch)
            
            con.commit()    
            batch.clear()  # Clear the batch for the next set of records
            
    
    # Insert any remaining records that are less than the batch size
    if batch:
        cur.executemany("insert into reviews_train(row_id, user_id, product_id, timestamp, rating) values (NULL,?,?,?,?)", batch)
        con.commit()
        batch.clear()  # Clear the batch for the next set of records

    con.close()

if __name__ == "__main__":
    
    save_to_database()