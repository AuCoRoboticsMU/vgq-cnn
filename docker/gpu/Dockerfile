FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV TZ=Europe/Dublin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Args
# `work_dir` must be an absolute path.
ARG work_dir=/root/Workspace

# Install `apt-get` deps.
RUN apt-get update && apt-get install -y \
        build-essential \
        python3 \
        python3-dev \
        python3-tk \
	python3-pip \
        python-opengl \
        curl \
        libsm6 \
        libxext6 \
	libspatialindex-dev \
        libglib2.0-0 \
        libxrender1 \
        wget \
	vim \
        unzip

# Required for easy_install to find right skimage version for Python 3.5.
RUN python3 -m pip install -U setuptools && python3 -m pip install --upgrade pip && pip3 --version

# Make working directory.
WORKDIR ${work_dir}

# Copy the library.
ADD docker/gqcnn.tar .

# This is because `python setup.py develop` skips install_requires (I think) and also because we want to explicitly use the GPU requirements.
RUN python3 -m pip install -r gqcnn/requirements/gpu_requirements.txt && \
	python3 -m pip install pandas

# Install the library in editable mode because it's more versatile (in case we want to develop or if users want to modify things)
# Keep the egg outside of the library in site-packages in case we want to mount the library (overwriting it) for development with docker
ENV PYTHONPATH ${work_dir}/gqcnn
WORKDIR /usr/local/lib/python3.5/site-packages/
RUN python3 ${work_dir}/gqcnn/setup.py develop --docker

# Move to the base library dir
WORKDIR ${work_dir}/gqcnn

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user
