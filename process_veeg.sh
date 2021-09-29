#!/bin/bash
docker run \
    -it \
    -v /home/idrael/git/VEEG_Event_Processor/data:/app/data \
    -v /home/idrael/git/VEEG_Event_Processor/results:/app/results \
rukrei/veeg_event_processor:0.1 \
python3 ./src/VEEG_processor.py