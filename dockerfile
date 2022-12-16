FROM ubuntu:18.04

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y python3.8 python3.8-distutils python3.8-dev
RUN apt-get install -y python3-pip

RUN python3.8 -m pip install --upgrade pip setuptools wheel 

RUN python3.8 -m pip install --no-cache-dir -r requirements.txt

ADD . /app

CMD [ "python3.8", "./cache.py"]