FROM continuumio/miniconda3

ADD src/VEEG_processor.py app/src/
ADD src/utils.py app/src/
ADD src/__init__.py app/src/
ADD src/VEEG_config.xlsx app/src/
ADD data/ app/data/

RUN pip install plotly plotly-express mne pandas numpy

WORKDIR /app

#CMD python /app/src/VEEG_processor.py
