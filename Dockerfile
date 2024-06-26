FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    git \
    cifs-utils \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils && \
    apt-get install -y software-properties-common vim curl unzip htop openssh-server wget less procps cmake libboost-all-dev

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir joblib numpy tqdm pillow scipy joblib matplotlib scikit-image argparse SimpleITK pyyaml pandas pydicom scikit-learn natsort opencv-python-headless wandb lmdb gudhi tensorboardX ninja pytest pot
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
