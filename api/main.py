from flask import Flask, request, jsonify
from service import APIService
from repository import APIRepository
from dotenv import load_dotenv
import os

load_dotenv()

db_conn_str = os.getenv("DB_CONN_STR")
engine_api_addr = os.getenv("ENGINE_API_ADDR")

app = Flask(__name__)

api_repository = APIRepository(
    db_conn_str=db_conn_str,
    engine_api_addr = engine_api_addr
)

api_service = APIService(
    api_repository=api_repository
)

@app.route('/list/<category_name>', methods=['GET'])
def get_top_k(category_name):
    # Check if the request content type is JSON
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    try:
        # 1. request data
        request_data = request.get_json()
        k = request_data["k"]
        
        # 2. transform the data fit to engine's data style
        category_code = api_service.get_category_code(category_name)
        product_ids = []
        ratings = []
        for review in request_data["user_reviews"]:
            product_ids.append(review['product_id_str'])
            ratings.append(float(review['rating']))
        product_ids = api_service.get_product_ids(category_code, product_ids)    
        
        # 3. request to engine
        recommended_product_ids = api_service.get_top_k(
            k = k,
            category_code = category_code,
            review_product_ids = product_ids,
            review_ratings = ratings
        )
        
        # 4. recommended_product_ids to recommended_product_id_strs
        recommended_product_id_strs = api_service.get_product_id_strs(
            category_code, recommended_product_ids)
       
        return jsonify(recommended_product_id_strs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(
        debug=True,
        host = "0.0.0.0",
        port=3005
    )
