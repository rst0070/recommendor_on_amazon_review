sudo docker container rm amazon_review_train
sudo docker build -t amazon_review_engine .
sudo docker run -d \
    --network="host" \
    --gpus all \
    --name amazon_review_train \
    --shm-size=50gb \
    -v ./parameters:/app/parameters \
    amazon_review_engine:latest