import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import torch
from models.transformer import TransformerReg
from train.configs import SysConfig, ExpConfig
from vector_db.product import ProductVectorDatabase

sys_config = SysConfig()
exp_config = ExpConfig()
vector_db = ProductVectorDatabase()

def get_param(category_code):
        
    path = os.path.join(
            os.path.dirname(__file__), '..', 'parameters', str(category_code)
        )
    min_error = 100.
    param_path = None
        
    for root, dirs, files in os.walk(path):
        for name in files:
            error_param = float(name.split('/')[-1].split('.')[0])
            if error_param < min_error:
                min_error = error_param
                param_path = os.path.join(root, name)
        
    return torch.load(param_path)
    
def get_product_ids(db_conn_str, category_code) -> list[int]:
    """
    returns products ids which is top 1000 (based on number of reviews) per category
    """
    engine = create_engine(db_conn_str)
    with engine.connect() as conn:
        res = conn.execute(
            text(
                f"""
                SELECT 
                    product_id
                FROM
                    product_info
                WHERE
                    category_code = '{category_code}'
                    and is_top_1000 = TRUE
                """
                )
            ).fetchall()
        
        result = []
        for row in res:
            result.append(row[0])
            
        return result # form: [0, 1, 2, ..]
    
def run(
    category_code, 
    db_conn_str
    ):
    
    # 1. get product embeddings
    model_param = get_param(category_code=category_code)
    model = TransformerReg(
        num_product=sys_config.num_product,
        embedding_dim=exp_config.embedding_size,
        num_transformer_block=exp_config.num_transformer_block,
        ffn_hidden=exp_config.ffn_hidden
    )
    model.load_state_dict(model_param)
    product_embeddings = model.product_embedding
    
    # 2. get product ids
    product_ids = get_product_ids(db_conn_str, category_code)
    product_embeddings = product_embeddings(
        torch.tensor(
            product_ids,
            dtype = torch.int32
        )
    )
    product_embeddings = product_embeddings.detach().numpy().tolist()
    print(f"number of product embeddings: {len(product_embeddings)}")
        
    # 3. store
    vector_db.store_batch(
        category_code=category_code,
        product_ids=product_ids,
        vectors = product_embeddings
    )

def check(category_code):
    
    result = vector_db.find_similar_vectors(
        category_code=category_code,
        query_vector=[1., 1., 1., 1., 1., 1., 1., 1.],
        k = 5
    )
    
    print(result)
    

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    db_conn_str = os.getenv('DB_CONN_STR')
    
    for category_code in range(1, 16):
        run(category_code, db_conn_str)
        check(category_code)