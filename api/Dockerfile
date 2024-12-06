FROM python:3.11.8-slim
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install curl -y
RUN python -m pip install --upgrade pip
# poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'
RUN curl -sSL https://install.python-poetry.org/ | python  
COPY ./ /app/
WORKDIR /app
RUN poetry install
CMD ["poetry", "run", "python", "main.py"]