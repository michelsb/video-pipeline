FROM openvino-base:v1.0

RUN pip install prometheus-client

ADD ./ ./

CMD [ "python3","-u","app.py" ]
