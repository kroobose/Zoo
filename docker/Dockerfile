FROM nvidia/cuda:11.4.3-cudnn8-devel-centos7
# MAINTAINER wangjun

#install python3.7.1
# from https://zhuanlan.zhihu.com/p/137288195
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
# 设置默认为python3
RUN set -ex \
    # 备份旧版本python
    && mv /usr/bin/python /usr/bin/python27 \
    && mv /usr/bin/pip /usr/bin/pip27 \
    # 配置默认为python3
    && ln -s /usr/local/python3/bin/python3.7 /usr/bin/python \
    && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip
# 修复因修改python版本导致yum失效问题
RUN set -ex \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/bin/yum \
    && sed -i "s#/usr/bin/python#/usr/bin/python2.7#" /usr/libexec/urlgrabber-ext-down \
    && yum install -y deltarpm
# 基础环境配置
RUN set -ex \
    # 修改系统时区为东八区
    && rm -rf /etc/localtime \
    && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && yum install -y vim \
    # 安装定时任务组件
    && yum -y install cronie
# 支持中文
RUN yum install kde-l10n-Chinese -y
RUN localedef -c -f UTF-8 -i zh_CN zh_CN.utf8
# 更新pip版本
RUN pip install --upgrade pip
ENV LC_ALL zh_CN.UTF-8

# install pytorch
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyyaml==3.13 scikit-image==0.14.1 numpy==1.15.4 scipy==1.1.0 torchvision==0.3.0 torch==1.1.0 tensorboardX==2.0 opencv-python==4.1.2.30 Cython==0.29.2 matplotlib==3.0.2
# miya added
RUN pip install protobuf==3.20.3 timm==0.4.5 torchvision==0.2.2 Pillow==6.1.0 PyYAML==5.1

# FROM nvcr.io/nvidia/pytorch:23.06-py3
# # RUN apt-get install -y gnupg2
# # RUN \
# #     # Update nvidia GPG key
# #     apt-key del 7fa2af80 && \
# #     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
# #     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# ENV TZ=Asia/Tokyo
# WORKDIR /workspace
# ENV LANG C.UTF-8
# ENV LANGUAGE en_US
# ENV DEBCONF_NOWARNINGS yes
# ENV PYTHONIOENCODING utf-8
# ENV DEBIAN_FRONTEND=noninteractive

# # package install
# RUN apt-get update && apt-get -y upgrade && apt-get install --no-install-recommends -y lsof git wget build-essential
# # autoremove
# RUN apt-get autoremove -y

# RUN apt-get install -y python3-pip
# RUN pip install --upgrade pip

# COPY requirements.txt .
# RUN pip install -r requirements.txt
