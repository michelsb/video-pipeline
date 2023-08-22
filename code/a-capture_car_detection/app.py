import os
import cv2
import imagezmq
import time
#import csv
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter
from util.preprocess import preprocess_image, image_to_tensor
from util.postprocess import postprocess
from openvino.runtime import Core

####### Device configuration #######
DETECTION_MODEL_PATH = os.environ.get("DETECTION_MODEL_PATH")
DEVICE = os.environ.get("DEVICE_TYPE")
####### Source configuration #######
SERVICE_ID = os.environ.get("SERVICE_ID")
CAMERA_CONFIG_ID = os.environ.get("CAMERA_CONFIG_ID")
CAMERA_CONFIG_SOURCE = os.environ.get("CAMERA_CONFIG_SOURCE")
#CAMERA_CONFIG_ROI_TOP_LEFT = eval(os.environ.get("CAMERA_CONFIG_ROI_TOP_LEFT")) #Formato "(500,500)"
#CAMERA_CONFIG_ROI_BOTTOM_RIGHT = eval(os.environ.get("CAMERA_CONFIG_ROI_BOTTOM_RIGHT"))
CAMERA_CONFIG_LOCAL = bool(os.environ.get("CAMERA_CONFIG_LOCAL"))
DEBUG = bool(os.environ.get("DEBUG"))
####### Teste em video local #######
CAMERA_CONFIG_FPS = int(os.environ.get("CAMERA_CONFIG_FPS"))
CAMERA_CONFIG_SECONDS = int(os.environ.get("CAMERA_CONFIG_SECONDS"))
######################################
CAMERA_CONFIG_RETRY_CONNECTION = 5

# Define metrics
CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
SENT_FRAME_BYTES = Summary('bytes_per_sent_frame', 'Number of bytes per sent frame')
SENT_FRAMES = Counter('sent_frames', 'Total of sent frames')
DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')

#DETECTED_CARS = Summary('detected_cars_per_frame', 'Number of detected cars per frame')


####### Define model config #######


def load_dnn_model():
    core = Core()
    det_ov_model = core.read_model(DETECTION_MODEL_PATH)
    if DEVICE != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    det_compiled_model = core.compile_model(det_ov_model, DEVICE)
    return det_compiled_model


def detect(image, model):
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, nc=3)
    return detections


def cut(results, source_image):
    boxes = results["det"]
    cuts = []
    for idx, (*box, conf, lbl) in enumerate(boxes):
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        c = (source_image[y1:y2,x1:x2], int(lbl))
        cuts.append(c)
    return cuts

def process_video(camera):
    
    frame_id = 0

    loop = CAMERA_CONFIG_SECONDS*CAMERA_CONFIG_FPS if CAMERA_CONFIG_LOCAL else float('inf')
    
    port = 5555
    sender = imagezmq.ImageSender("tcp://*:{}".format(port), REQ_REP=False)
    
    vehicle_detection = load_dnn_model()

    while(frame_id < loop):
        
        sucess_capture, frame = camera.read()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Aplicando ROI
        #frame = frame[CAMERA_CONFIG_ROI_TOP_LEFT[1]:CAMERA_CONFIG_ROI_BOTTOM_RIGHT[1],
        #              CAMERA_CONFIG_ROI_TOP_LEFT[0]:CAMERA_CONFIG_ROI_BOTTOM_RIGHT[0]]
        
        if not sucess_capture:
            camera.release()
            time.sleep(CAMERA_CONFIG_RETRY_CONNECTION)
            break

        CAPTURED_FRAME_BYTES.observe(frame.nbytes)
        CAPTURED_FRAMES.inc()
        
        init_detection = datetime.now().timestamp()
        
        detections = detect(frame.copy(), vehicle_detection)[0]
        
        all_vehicles = cut(detections, frame)
        
        end_detection = datetime.now().timestamp()
        
        detection_time = end_detection - init_detection
        
        DETECTION_TIME.observe(detection_time)
        #DETECTED_CARS.observe(len(all_vehicles))

        if DEBUG: print("[DEBUG] Time:",end_detection,"| Frame ID:",frame_id,"| Detection Time:",detection_time,"| Number of detected cars:",len(all_vehicles))
        
        vehicle_id = 0

        for vehicle,c in all_vehicles:
            json_string = f'{{"frame_id": "{frame_id}", "vehicle_id": "{vehicle_id}", "vehicle_class": "{c}"}}'
            sender.send_image(json_string, vehicle)
            SENT_FRAME_BYTES.observe(vehicle.nbytes)
            SENT_FRAMES.inc()
            vehicle_id +=1
         
        
        frame_id+=1

        
    sender.zmq_socket.close()


if __name__ == "__main__":
    # Start up the server to expose the metrics.
    start_http_server(8000)
    #if DEBUG:
        #header = ['name', 'area', 'country_code2', 'country_code3']

    while True:
        camera = cv2.VideoCapture( CAMERA_CONFIG_SOURCE )
        if camera.isOpened():
            process_video(camera)
        else:
            time.sleep(CAMERA_CONFIG_RETRY_CONNECTION)
        #break
