import os
import cv2
import sys
import imagezmq
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

PREVIOUS_MODULE = os.environ.get("PREVIOUS_MODULE")

def sendImagesToWeb():
    #receiver = imagezmq.ImageHub(open_port='tcp://capture_detection:5555', REQ_REP = False)
    #receiver = imagezmq.ImageHub()
    hostname = PREVIOUS_MODULE
    #hostname = "172.20.0.2"
    port = 5555
    receiver = imagezmq.ImageHub("tcp://{}:{}".format(hostname, port), REQ_REP=False)
    while True:
        camName, frame = receiver.recv_image()
        jpg = cv2.imencode('.jpg', frame)[1]
        yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'
   
@Request.application
def application(request):
    return Response(sendImagesToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    run_simple('0.0.0.0', 4000, application)