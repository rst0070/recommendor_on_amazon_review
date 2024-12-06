from flask import Flask, request, jsonify
from models.transformer import TransformerReg
from train.configs import SysConfig, ExpConfig
from vector_db.product import ProductVectorDatabase
import torch
import os

db_conn_str = ""
path_param_dir = ""

app = Flask(__name__)

class Service:
    
    sys_config = SysConfig()
    exp_config = ExpConfig()
    models:dict[int, TransformerReg] = {}
    product_db = ProductVectorDatabase()
    
    @classmethod
    def _get_param(cls, category_code):
        
        path = os.path.join(
                os.path.dirname(__file__), 'parameters', str(category_code)
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

    @classmethod
    def get_user_vector(
        cls, 
        category_code,
        product_ids:list[int],
        ratings:list[int]
        ) -> list[int]:
        # 1. get model
        # 2. pass the parameters
        # 3. return
        if category_code not in cls.models.keys():
            
            cls.models[category_code] = TransformerReg(
                num_product=cls.sys_config.num_product,
                embedding_dim=cls.exp_config.embedding_size,
                num_transformer_block=cls.exp_config.num_transformer_block,
                ffn_hidden=cls.exp_config.ffn_hidden
            )
            
            cls.models[category_code].load_state_dict(
                cls._get_param(category_code)
            )
            
        with torch.no_grad():
            
            user_vector = cls.models[category_code].forward(
                target_product = None,
                products = torch.tensor([product_ids, product_ids], dtype=torch.int32),
                ratings = torch.tensor([ratings, ratings], dtype=torch.int32),
                is_training=False
            )
            
            return user_vector[0].numpy().tolist()
    
    @classmethod
    def get_top_k(
        cls,
        category_code,
        k,
        user_vector
        ) -> list[int]:
        """
        Get product ids, similar to user vector
        """
        assert type(category_code) is int
        
        return cls.product_db.find_similar_vectors(
            category_code=category_code,
            query_vector=user_vector,
            k = k
        )


@app.route('/<category_code>', methods=['GET'])
def prediction(category_code):
    """
    
    """
    # Check if the request content type is JSON
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    request_data = request.get_json()
    category_code = int(category_code)
    print(category_code)
    
    # 1. request data
    k = int(request_data["k"])
    review_product_ids = request_data["review_product_ids"]
    review_ratings = [int(rating) for rating in request_data["review_ratings"]]
        
    # 2. get user vector
    user_vector = Service.get_user_vector(category_code, review_product_ids, review_ratings)
    print(user_vector)
    # 3. get similar products (integer ids)
    predicts = Service.get_top_k(category_code, k, user_vector)
    
    result = {
        "product_ids": predicts
    }
    print(result)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(
        debug=True,
        host = "0.0.0.0",
        port=3006
        )
