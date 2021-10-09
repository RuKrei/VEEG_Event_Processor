FROM continuumio/miniconda3:4.10.3

ADD ./src/VEEG_processor.py /app/src/
ADD ./requirements.txt /app/src/
ADD ./data/* /app/data/

RUN mkdir -p /app/results
RUN pip install -r /app/src/requirements.txt

WORKDIR /app

