from typing import Tuple
from pyspark import SparkContext, RDD
import os
from tqdm import trange

cwd = os.getcwd()

def load_datasets():

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
                    )['full']

    item_meta_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_meta_Home_and_Kitchen",
                        split="full",
                        trust_remote_code=True
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

def store_rating_data():
    
    review_dataset, _ = load_datasets()
    
    
    user_dir = os.path.join(cwd, "data/rating/user")
    product_dir = os.path.join(cwd, "data/rating/product")
    
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(product_dir, exist_ok=True)
    
    for r_idx in trange(0, len(review_dataset)):
        
        review = review_dataset[r_idx]
        
        user_id = review['user_id']
        product_id = review['parent_asin']
        rating = float(review['rating'])
        
        with open(os.path.join(user_dir, user_id), "a") as user_file:
            user_file.write(f"{product_id},{rating}\n")
            user_file.close()
            
        with open(os.path.join(product_dir, product_id), "a") as product_file:
            product_file.write(f"{user_id},{rating}\n")
            product_file.close()

if __name__ == "__main__":
    
    store_rating_data()
            