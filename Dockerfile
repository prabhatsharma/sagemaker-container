# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM nvidia/cuda:9.0-base-ubuntu16.04
# FROM ubuntu:16.04
# FROM smtf

LABEL version="1.0"
LABEL Author="Prabhat Sharma <prabhsha@amazon.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \
        curl \
        git \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        libcudnn7-dev=7.0.5.15-1+cuda9.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        wget \
        vim \
        nginx \
        iputils-ping \
        && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local/cuda-9.0/lib64/ -type f -name 'lib*_static.a' -not -name 'libcudart_static.a' -delete && \
    rm /usr/lib/x86_64-linux-gnu/libcudnn_static_v7.a

# RUN apt-get -y update && apt-get install -y --no-install-recommends \
#     apt-utils \
#     wget \
#     python \
#     nginx \
#     ca-certificates \
#     python3.5 \
#     python3-pip \
#     python-pip \
#     libgtk2.0 \
#     protobuf-compiler python-pil python-lxml python-tk \
#     && rm -rf /var/lib/apt-get/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        numpy \
        scipy \
        sklearn \
        pandas \
        h5py

RUN pip install numpy tensorflow-serving-api==1.5


# Set up grpc
RUN pip install enum34 futures mock six && \
    pip install --pre 'protobuf>=3.0.0a3' && \
    pip install -i https://testpypi.python.org/simple --pre grpcio

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.8.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Configure the build for our CUDA configuration.
ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.7,6.1
ENV TF_CUDA_VERSION=9.0
ENV TF_CUDNN_VERSION=7
ENV CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu

ENV TF_SERVING_VERSION=1.5.0

# Download TensorFlow Serving
RUN cd / && git clone --recurse-submodules https://github.com/tensorflow/serving && \
  cd serving && \
  git checkout $TF_SERVING_VERSION

# Configure Tensorflow to use the GPU
WORKDIR /serving
RUN git clone --recursive https://github.com/tensorflow/tensorflow.git && \
  cd tensorflow && \
  git checkout v$TF_SERVING_VERSION && \
  tensorflow/tools/ci_build/builds/configured GPU

# Build TensorFlow Serving and Install it in /usr/local/bin
WORKDIR /serving
RUN bazel build -c opt --config=cuda \
    --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
    --crosstool_top=@local_config_cuda//crosstool:toolchain \
    tensorflow_serving/model_servers:tensorflow_model_server && \
    cp bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/local/bin/ && \
    bazel clean --expunge

# cleaning up the container
RUN rm -rf /serving && \
    rm -rf /bazel


# Install object_detection dependencies
RUN apt-get -y install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev 

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git 

RUN echo alias python=python3 >> ~/.bashrc
# RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#     wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
#     /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#     rm ~/anaconda.sh

# ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip
RUN pip install -U setuptools
RUN pip install pandas tensorflow Cython pillow lxml matplotlib

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install pandas tensorflow Cython pillow lxml matplotlib

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN rm -rf /root/.cache

# Copy test data - Used only during testing. Not required in final build
COPY ./local_test/test_dir /opt/ml

# Set up the program in the image
COPY tensorflow /opt/program

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
RUN python object_detection/builders/model_builder_test.py

# Downloading a COCO-pretrained Model for Transfer Learning
RUN wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz \
    && tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
# RUN pwd && ls
WORKDIR /opt/program

