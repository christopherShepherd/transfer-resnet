# syntax=docker/dockerfile:1
FROM ubuntu:20.04

RUN apt-get -y update && \
    apt-get install -y python3-pip

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

COPY ./transfer_resnet /transfer_resnet

WORKDIR /transfer_resnet

ENTRYPOINT ["python3", "transfer_resnet.py"]
