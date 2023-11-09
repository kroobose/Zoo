FROM nvidia/cuda:11.4.3-cudnn8-devel-centos7

#install python3.7.1
RUN set -ex \
    && yum install -y wget tar libffi-devel zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make initscripts \
    && wget https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tgz \
    && tar -zxvf Python-3.7.1.tgz \
    && cd Python-3.7.1 \
    && ./configure prefix=/usr/local/python3 \
    && make \
    && make install \
    && make clean \
    && rm -rf /Python-3.7.1* \
    && yum install -y epel-release \
    && yum install -y python-pip \
    && yum install -y mesa-libGL.x86_64 \
    && yum install -y libSM-1.2.2-2.el7.x86_64

# Set default python to python3
RUN set -ex \
    && mv /usr/bin/python /usr/bin/python27 \
    && mv /usr/bin/pip /usr/bin/pip27 \
    && ln -s /usr/local/python3/bin/python3.7 /usr/bin/python \
    && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip

# Fix yum after python update
RUN set -ex \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/bin/yum \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/libexec/urlgrabber-ext-down \
    && yum install -y deltarpm

# Basic environment configuration
RUN set -ex \
    && rm -rf /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && yum install -y vim \
    && yum -y install cronie

# Support Japanese
RUN yum install -y kde-l10n-Japanese glibc-common
RUN localedef -c -f UTF-8 -i ja_JP ja_JP.utf8

# Update pip version
RUN pip install --upgrade pip
ENV LC_ALL ja_JP.UTF-8

# Install other python packages
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyyaml==3.13 scikit-image==0.14.1 numpy==1.15.4 scipy==1.1.0 torchvision==0.3.0 torch==1.1.0 tensorboardX==2.0 opencv-python==4.1.2.30 Cython==0.29.2 matplotlib==3.0.2

# Additional packages by miya
RUN pip install protobuf==3.20.3 timm==0.4.5 torchvision==0.2.2 Pillow==6.1.0 PyYAML==5.1
