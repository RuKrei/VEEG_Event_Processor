FROM continuumio/miniconda3:4.10.3

ADD ./src/VEEG_processor.py /app/src/
ADD ./src/VEEG_config.xlsx /app/src/
ADD ./requirements.txt /app/src/
ADD ./data/*.edf /app/data/

RUN mkdir -p /app/results
RUN pip install -r /app/src/requirements.txt

WORKDIR /app

