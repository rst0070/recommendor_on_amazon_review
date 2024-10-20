FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update

COPY ./ /app/
WORKDIR /app

RUN pip install pip --upgrade
RUN pip install torch -r ./requirements.txt

CMD ["python",  "run_train.py"]