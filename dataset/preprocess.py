from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tqdm import tqdm
import json
import random

dataset_path_list = [
    "All_Beauty.jsonl",
    "Amazon_Fashion.jsonl",
    "Appliances.jsonl",
    "Arts_Crafts_and_Sewing.jsonl",
    "Automotive.jsonl",
    "Baby_Products.jsonl",
    "Beauty_and_Personal_Care.jsonl",
    "Books.jsonl",
    "CDs_and_Vinyl.jsonl",
    "Cell_Phones_and_Accessories.jsonl",
    "Clothing_Shoes_and_Jewelry.jsonl",
    "Digital_Music.jsonl",
    "Electronics.jsonl",
    "Gift_Cards.jsonl",
    "Grocery_and_Gourmet_Food.jsonl",
    "Handmade_Products.jsonl",
    "Health_and_Household.jsonl",
    "Home_and_Kitchen.jsonl",
    "Industrial_and_Scientific.jsonl",
    "Kindle_Store.jsonl",
    "Magazine_Subscriptions.jsonl",
    "Movies_and_TV.jsonl",
    "Musical_Instruments.jsonl",
    "Office_Products.jsonl",
    "Patio_Lawn_and_Garden.jsonl",
    "Pet_Supplies.jsonl",
    "Software.jsonl",
    "Sports_and_Outdoors.jsonl",
    "Subscription_Boxes.jsonl",
    "Tools_and_Home_Improvement.jsonl",
    "Toys_and_Games.jsonl",
    "Unknown.jsonl",
    "Video_Games.jsonl"
]

def upload_to_db(
        engine,
        category_code: int,
        batch: list[dict]
    ):
    
    target_table = random.choices(
            ["review_train","review_test","review_valid"],
            [8.0,1.0,1.0],
            k = 1
        )[0]
    
    with engine.begin() as conn:
        
        # insert review data
        conn.execute(
            text(
            f"""
            INSERT INTO {target_table}( category_code, review_id, user_id, product_id, rating )
            VALUES( {category_code}, :review_id, :user_id, :product_id, :rating )
            ON CONFLICT(category_code, review_id) DO NOTHING
            """),
            batch
        )
        
        if target_table != "review_train":
            return
        
        # insert review reference (per user) data
        conn.execute(
            text(
            "INSERT INTO review_train_reference( category_code, user_id, product_ids, ratings ) " +
            f"VALUES( {category_code}" + ", :user_id, '{ :product_id }', '{ :rating }' )" + 
            f"""
            ON CONFLICT (category_code, user_id)
            DO UPDATE 
            SET 
                product_ids = review_train_reference.product_ids    || :product_id, 
                ratings     = review_train_reference.ratings        || :rating
            WHERE
                review_train_reference.category_code = {category_code} and 
                review_train_reference.user_id = :user_id
            """
            ),
            batch
        )

def main(batch_size):
    
    engine = create_engine('postgresql+psycopg2://rootuser:rootpass@localhost:3011/amazon_reviews_2023', echo=False)
    
    for category_code, dataset_path in enumerate(dataset_path_list):
        category = dataset_path.split('.')[0]
        print(f"Storing {category} into category code {category_code}..")
        
        user_ids = {}
        product_ids = {}
        
        user_id_new = 0
        product_id_new = 0
        
        with open(dataset_path, "r") as dataset:
            
            count = 0
            batch = []
            
            for review_id, line in enumerate(tqdm(dataset)):
                
                count += 1
                json_data = json.loads(line)
                
                try:
                    rating = int(json_data['rating'])
                except:
                    #print(json_data['rating'][0])
                    continue
                
                try:
                    user_id_str = json_data['user_id']
                    if user_id_str not in user_ids.keys():
                        user_ids[user_id_str] = user_id_new
                        user_id_new += 1
                        
                    user_id = user_ids[user_id_str]
                except:
                    continue
                
                try:
                    product_id_str = json_data['parent_asin']
                    if product_id_str not in product_ids.keys():
                        product_ids[product_id_str] = product_id_new
                        product_id_new += 1
                        
                    product_id = product_ids[product_id_str]
                except:
                    continue
                
                batch.append(
                        {
                            "review_id": review_id, 
                            "user_id" : user_id, 
                            "product_id":product_id, 
                            "rating":rating
                        }
                    )
                
                if count == batch_size:
                    upload_to_db(
                        engine,
                        category_code,
                        batch
                    )
                    count = 0
                    batch.clear()
                    
            if count > 0:
                upload_to_db(
                    engine,
                    category_code,
                    batch
                )
                batch.clear()
                    
        print(f"Category code {category_code} is completed!")
            
        

if __name__ == "__main__":
    main(batch_size=1000)