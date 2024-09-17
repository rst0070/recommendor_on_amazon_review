import sqlite3
from exp.exp1 import Exp

def main():
    
    db_connection = sqlite3.connect("reviews.db")
    exp = Exp(db_connection=db_connection)
    
    print(exp.get_pearson_coeff('B09XWYG6X1', 'B0BXDLF8TW'))
    
if __name__ == "__main__":
    main()