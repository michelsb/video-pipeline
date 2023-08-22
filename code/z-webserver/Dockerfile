FROM pytorch1.13.1-cuda11.6-cudnn8-runtime

ADD ./ ./

RUN pip install -U Werkzeug
RUN pip install prometheus-client

CMD [ "python3", "app.py" ]
