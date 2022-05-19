#!/bin/bash

rm docker/gqcnn.tar
tar -cvf docker/gqcnn.tar --exclude="models" --exclude="analysis" ../gqcnn

docker build -t gqcnn:gpu -f docker/gpu/Dockerfile \
		--build-arg USER_ID=$(id -u) \
		--build-arg GROUP_ID=$(id -g) .

