FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update

COPY ./ /app/
WORKDIR /app

RUN pip install pip --upgrade
RUN pip install torch -r ./requirements.txt