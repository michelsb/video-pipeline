import os
import cv2
import imagezmq
import json
from ultralytics import YOLO
import random

#from torchvision import transforms
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter
from util.preprocess import preprocess_image, image_to_tensor
from util.postprocess import postprocess
from openvino.runtime import Core

####### Device configuration #######
# DETECTION_MODEL_PATH = os.environ.get("DETECTION_MODEL_PATH")
# DEVICE = os.environ.get("DEVICE_TYPE")
# ####### Source configuration #######
# PREVIOUS_MODULE = '127.0.0.1'
# DEBUG = bool(os.environ.get("DEBUG"))

# # Define metrics
# CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
# CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
# SENT_FRAME_BYTES = Summary('bytes_per_sent_frame', 'Number of bytes per sent frame')
# SENT_FRAMES = Counter('sent_frames', 'Total of sent frames')
# DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')

#DETECTED_FRAME_BYTES = Summary('bytes_per_detected_frame', 'Number of bytes per detected plate')
#DETECTED_PLATES = Summary('detected_plates_per_frame', 'Number of detected plates per frame')
#DROPPED_PLATES = Summary('dropped_plates', 'Number of dropped plates per frame')
DETECTION_MODEL_PATH = './best_openvino_model'
DEVICE = os.environ.get("DEVICE_TYPE")
####### Source configuration #######
PREVIOUS_MODULE = '127.0.0.1'
DEBUG = False

# Define metrics
CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
SENT_FRAME_BYTES = Summary('bytes_per_sent_frame', 'Number of bytes per sent frame')
SENT_FRAMES = Counter('sent_frames', 'Total of sent frames')
DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')


model = YOLO(DETECTION_MODEL_PATH)
if __name__ == "__main__":    

    start_http_server(8001)    
    
    port = 5555
    port_2 = 5556 # fiz isso não estou usando o docker
    image_hub = imagezmq.ImageHub("tcp://{}:{}".format(PREVIOUS_MODULE, port), REQ_REP=False)
    sender = imagezmq.ImageSender("tcp://*:{}".format(port_2), REQ_REP=False)


    #allFramesBytes = []
    #boundBoxesH = []
    #boundBoxesW = []

    #all_detection_time = []
    #all_filter_time = []
    #filterNotDiscart = 0

    while True:

        json_string, frame = image_hub.recv_image()

        CAPTURED_FRAME_BYTES.observe(frame.nbytes)
        CAPTURED_FRAMES.inc()
        
        json_object = json.loads(json_string)
        
        init_detection = datetime.now().timestamp()
        all_plates = []
        results = model.predict(source=frame, conf=0.4, imgsz=640, classes = [0], verbose = False)
        for result in results:
            for detection in result.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                all_plates.append((frame[y1:y2, x1:x2].copy(), cls))
                classe = cls.item()
                if random.random() < 0.5: #pra simular o descarte por qualidade
                    classe += 5

                all_plates.append((frame[y1:y2, x1:x2].copy(), classe))

        end_detection = datetime.now().timestamp()

        detection_time = end_detection - init_detection
        
        #all_detection_time.append(detection_time)
        DETECTION_TIME.observe(detection_time)
        #DETECTED_PLATES.observe(len(all_plates))

        filterNotDiscart = 0

        for plate,c in all_plates:    
            c = int(c)  
            plate_id = 0

            json_object["plate_id"] = plate_id

            #DETECTED_FRAME_BYTES.observe(plate.nbytes)            

            json_object["plate_class"] = c
            
            if DEBUG: print("[DEBUG] Time:",datetime.now().timestamp(),"| Frame ID:",json_object["frame_id"],"| Vehicle ID:", json_object["vehicle_id"],"| Detection Time:",detection_time,"| Number of detected plates: ",len(all_plates),"| Plate ID:",plate_id,"| Plate Class: ",int(c))

            if c in [0,1,2,4,5]:
                filterNotDiscart += 1
                json_object["filter_class"] = c in [0,4]
                new_json_string = json.dumps(json_object)
                print(new_json_string)
                SENT_FRAME_BYTES.observe(plate.nbytes)
                SENT_FRAMES.inc()
                sender.send_image(new_json_string, plate)           
            
            plate_id +=1

        #DROPPED_PLATES.observe(len(all_plates)-filterNotDiscart)

    sender.zmq_socket.close()
