import sqlite3
from preprocess.pearson_sim import run

if __name__ == "__main__":
    
    db_connection = sqlite3.connect("reviews.db")
    run(db_connection)
            