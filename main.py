import sqlite3
from exp.exp1 import Exp
import math
from tqdm import tqdm

def test():
    """
    
    """
    db_connection = sqlite3.connect("reviews.db")
    exp = Exp(db_connection=db_connection)
    
    cursor = db_connection.cursor()
    test_set = cursor.execute("SELECT product_id, user_id, rating FROM reviews_test").fetchall()
    
    error = 0.0
    count = 0
    
    for product_id, user_id, rating in tqdm(test_set):
        
        tmp = exp.test(user_id, product_id)
        if tmp is not None:
            error += math.sqrt((tmp - rating) ** 2) / rating
            count += 1
            
    error = error / count
    print(f"error: {error}, count = {count}")

def main():
    """
    1. connect db
    2. get test data
    3. guess the right ans of test data using exp model
    4. using metric, get the score of the model
    
    """
    test()
    
    
if __name__ == "__main__":
    main()