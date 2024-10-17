import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from tqdm import tqdm
import random

def create_schema(engine:Engine):
    """
    Creates schema for DB
    """
    with engine.begin() as conn:
        
        conn.execute(
            text(
                """
                CREATE TABLE product_info(
                    product_id_int integer primary key,
                    product_id TEXT unique
                )
                """
            )
        )
        
        conn.execute(
                text(
                    """
                    CREATE TABLE reviews_train(
                        row_id integer primary key,
                        user_id TEXT,
                        product_id TEXT,
                        rating REAL,
                        product_id_int integer,
                        foreign key(product_id_int) references product_info(product_id_int)
                    );
                    """
                )
            )
        
        conn.execute(
                text(
                    """
                    CREATE TABLE reviews_validation(
                        row_id integer primary key,
                        user_id TEXT,
                        product_id TEXT,
                        rating REAL,
                        product_id_int integer,
                        foreign key(product_id_int) references product_info(product_id_int)
                    );
                    """
                )
            )
        
        conn.execute(
                text(
                    """
                    CREATE TABLE reviews_test(
                        row_id integer primary key,
                        user_id TEXT,
                        product_id TEXT,
                        rating REAL,
                        product_id_int integer,
                        foreign key(product_id_int) references product_info(product_id_int)
                    );
                    """
                )
            )
        

def insert_reviews(engine:Engine, review_batch:list[dict]):
    """
    Inserts into reviews table, using review batch it gets.
    """
    branch = random.randint(1, 10)
    table_name = "reviews_"
            
    if branch == 1:
        table_name += 'test'
    elif branch == 2:
        table_name += 'validation'
    else:
        table_name += 'train'
        
    with engine.begin() as conn:
        conn.execute(
            text(
            f"""
            INSERT INTO {table_name}(user_id, product_id, rating)
            VALUES(:user_id, :product_id, :rating) 
            """
            ),
            review_batch
        )
        

        
    
def load_data_to_db(
    engine:Engine,
    cache_dir:str,
    batch_size:int):
    """
    cache_dir - dir path for caching datasets.load_dataset
    batch_size - batch size when loading to db
    """
    
    create_schema(engine)
    
    from datasets import load_dataset
    
    review_dataset = load_dataset(
                        "McAuley-Lab/Amazon-Reviews-2023",
                        "raw_review_Home_and_Kitchen",
                        trust_remote_code=True,
                        cache_dir=cache_dir,
                        streaming=False
                    )['full']
    """
    
    """
    
    batch = []
    
    for review in tqdm(review_dataset):
        
        batch.append(
                {
                    'user_id':review['user_id'],
                    'product_id':review['parent_asin'],
                    'rating':review['rating']
                }
            )
        
        # Once the batch size is reached, insert into the database
        if len(batch) == batch_size:
            insert_reviews(engine, batch)
            batch.clear()  # Clear the batch for the next set of records
            
    
    # Insert any remaining records that are less than the batch size
    if batch:
        insert_reviews(engine, batch)
        batch.clear()
        
        
def generate_int_id(engine:Engine):
    """
    1. Create index on reviews for user_id and product_id
    2. insert product_id to make product_id_int
    3. create index on product_info table for product_id
    4. update reviews table
    5. create index on reviews table for product_id_int
    """
    
    reviews_table_names = ['reviews_test', 'reviews_validation', 'reviews_train']
    
    # -------------------- Create index on reviews for user_id and product_id
    #
    with engine.begin() as conn:
        for table in tqdm(reviews_table_names, desc="Create index on reviews for user_id and product_id"):
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_user_id on {table} (user_id)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_product_id on {table} (product_id)"))    
            
    # -------------------- insert product_id to make product_id_int
    #
    with engine.begin() as conn:
        for table in tqdm(reviews_table_names, desc="insert product_id to make product_id_int"):
            conn.execute(
                text(
                f"""            
                INSERT INTO product_info(product_id)
                SELECT
                    DISTINCT product_id
                FROM
                    {table}
                WHERE
                    product_id NOT IN (
                        SELECT
                            product_id
                        FROM
                            product_info
                    )
                """
                )
            )
    # -------------------- create index on product_info table for product_id
    #
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_product_info_product_id ON product_info (product_id)"))
    
    print("create index on product_info table for product_id: done.")
    
    # -------------------- update reviews table
    #
    with engine.begin() as conn:
        for table in tqdm(reviews_table_names, desc="update reviews table"):
            conn.execute(
                text(
                f"""
                UPDATE
                    {table}
                SET
                    product_id_int = (
                            SELECT 
                                product_id_int 
                            FROM 
                                product_info 
                            WHERE 
                                {table}.product_id = product_info.product_id
                        )
                """
                )
            )
    
    # -------------------- create index on reviews table for product_id_int
    #
    with engine.begin() as conn:
        for table in tqdm(reviews_table_names, desc="create index on reviews table for product_id_int"):
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table}_product_id_int on {table} (product_id_int)"))
        
    
if __name__ == "__main__":
    import os.path as path
    
    db_path = path.realpath(path.join(path.dirname(__file__), "data/database.db"))
    sqlite_conn_str = f"sqlite:////{db_path}"
    
    cache_dir_path = path.join(path.dirname(__file__), 'cache')
    
    engine = create_engine(sqlite_conn_str)
    
    # load review data
    load_data_to_db(
        engine = engine, 
        cache_dir=cache_dir_path,
        batch_size=1000
        )
    # calculate some information from db
    generate_int_id(engine)
    