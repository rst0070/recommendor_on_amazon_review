sudo docker container rm amazon_review_train
sudo docker build -t amazon_review_engine .
sudo docker run \
    --gpus all \
    --name amazon_review_train \
    --shm-size=50gb \
    -v ./data/database.db:/app/data/database.db \
    -v ./parameters:/app/parameters \
    amazon_review_engine:latest