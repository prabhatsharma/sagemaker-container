# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

# FROM tensorflow/tensorflow:1.7.0-devel-gpu
# FROM tensorflow/tensorflow:1.7.0-gpu-py3
FROM tensorflow/tensorflow:latest-gpu
# FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
# FROM ubuntu:16.04
# FROM smtf

LABEL version="1.0"
LABEL Author="Prabhat Sharma <prabhsha@amazon.com>"

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    apt-utils \
    wget \
    python \
    nginx \
    ca-certificates \
    python2.7 \
    python3.5 \
    python3-pip \
    python-pip \
    libgtk2.0 \
    protobuf-compiler python-pil python-lxml python-tk \
    && rm -rf /var/lib/apt-get/lists/*

# RUN apt install --allow-downgrades -y libcudnn7-dev=7.0.5.15-1+cuda9.1 libcudnn7=7.0.5.15-1+cuda9.1
# Install object_detection dependencies
RUN apt-get -y install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev 

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git python3-pip 

RUN echo alias python=python3 >> ~/.bashrc
# RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#     wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
#     /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#     rm ~/anaconda.sh

# # ENV PATH /opt/conda/bin:$PATH
# RUN pip install --upgrade pip
RUN pip install -U setuptools
RUN pip install pandas Cython pillow lxml matplotlib
RUN pip install tensorflow-gpu
# RUN pip install Cython pillow lxml
# RUN pip install matplotlib

# RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install pandas Cython pillow lxml matplotlib
RUN pip3 install tensorflow-gpu

#RUN pip3 install pandas tensorflow-gpu==1.5 Cython pillow lxml matplotlib
# RUN pip3 install Cython pillow lxml
# RUN pip3 install matplotlib



# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}:/usr/local/cuda-9.0/bin"
# ENV LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"

RUN rm -rf /root/.cache

# Copy test data - Used only during testing. Not required in final build
COPY ./local_test/test_dir /opt/ml

# Set up the program in the image
COPY tensorflow /opt/program

#COPY lib/libcuda.so.1 /usr/lib/libcuda.so.1
#COPY lib/libnvidia-fatbinaryloader.so.384.111 /usr/lib/libnvidia-fatbinaryloader.so.384.111
# COPY lib/libnvidia-fatbinaryloader.so.390.46 /usr/lib/libnvidia-fatbinaryloader.so.390.46

WORKDIR /opt/program

# RUN pwd && ls -lh
# Install tensorflow models
RUN git clone --depth 1 https://github.com/tensorflow/models && cd models/research
# RUN git clone https://github.com/tensorflow/models && cd models/research

WORKDIR /opt/program/models/research

# Install COCO API
RUN git clone --depth 1 https://github.com/cocodataset/cocoapi.git
WORKDIR /opt/program/models/research/cocoapi/PythonAPI
RUN make && cp -r pycocotools /opt/program/models/research/

# Protobuf Compilation
# RUN protoc --version
WORKDIR /opt/program/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# Add Libraries to PYTHONPATH
# RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# RUN echo 1
ENV PYTHONPATH=$PYTHONPATH:/opt/program/models/research:/opt/program/models/research/slim

# Testing the Installation
# RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim && env && python object_detection/builders/model_builder_test.py
RUN echo 4
RUN env
RUN python --version
RUN python3 --version
# RUN python3 object_detection/builders/model_builder_test.py

# Downloading a COCO-pretrained Model for Transfer Learning
RUN wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz \
    && tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
# RUN pwd && ls
WORKDIR /opt/program

