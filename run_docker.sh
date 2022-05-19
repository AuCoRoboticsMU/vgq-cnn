#!/bin/bash

DATA_PATH='INCLUDE_YOUR_DATA_DIR'
EXPER_PATH='INCLUDE_YOUR_DATA_DIR'

xhost +local:root

docker run -it \
-v ${PWD}:${PWD} \
-v $DATA_PATH:/data \
-v $EXPER_PATH:/results \
-w ${PWD} \
-e DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
--gpus all \
--name gqcnn_gpu \
gqcnn:gpu

xhost -local:root

docker stop gqcnn_gpu
docker rm gqcnn_gpu
