#FROM python:3.10
#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ADD ./ ./

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg libsm6 libxext6

RUN pip install -r requirements.txt
