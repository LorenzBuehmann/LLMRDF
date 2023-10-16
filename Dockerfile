# Dockerfile

# pull the official docker image
FROM python:3.11.5-slim-bookworm

# set work directory
#WORKDIR /app

RUN apt-get update &&\
    apt-get install --no-install-recommends --yes build-essential

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pysqlite3-binary

# copy project
COPY . .

