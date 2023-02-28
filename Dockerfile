FROM python:3.9.13 as base

RUN apt-get -y update

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .