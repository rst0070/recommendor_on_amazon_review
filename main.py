import sqlite3
from exp.exp1 import Exp

def test():
    """
    
    """
    pass

def main():
    """
    1. connect db
    2. get test data
    3. guess the right ans of test data using exp model
    4. using metric, get the score of the model
    
    """
    
    db_connection = sqlite3.connect("reviews.db")
    exp = Exp(db_connection=db_connection)
    
    print(exp.get_pearson_coeff('B09XWYG6X1', 'B0BXDLF8TW'))
    
if __name__ == "__main__":
    main()