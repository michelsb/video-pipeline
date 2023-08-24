import os
import cv2
import imagezmq
import time
from ultralytics import YOLO
#import csv
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter
from util.preprocess import preprocess_image, image_to_tensor
from util.postprocess import postprocess
from openvino.runtime import Core

# ####### Device configuration #######
# DETECTION_MODEL_PATH = os.environ.get("DETECTION_MODEL_PATH")
# DEVICE = os.environ.get("DEVICE_TYPE")
# ####### Source configuration #######
# SERVICE_ID = os.environ.get("SERVICE_ID")
# CAMERA_CONFIG_ID = os.environ.get("CAMERA_CONFIG_ID")
# CAMERA_CONFIG_SOURCE = os.environ.get("CAMERA_CONFIG_SOURCE")
# #CAMERA_CONFIG_ROI_TOP_LEFT = eval(os.environ.get("CAMERA_CONFIG_ROI_TOP_LEFT")) #Formato "(500,500)"
# #CAMERA_CONFIG_ROI_BOTTOM_RIGHT = eval(os.environ.get("CAMERA_CONFIG_ROI_BOTTOM_RIGHT"))
# CAMERA_CONFIG_LOCAL = bool(os.environ.get("CAMERA_CONFIG_LOCAL"))
# DEBUG = bool(os.environ.get("DEBUG"))
# ####### Teste em video local #######
# CAMERA_CONFIG_FPS = int(os.environ.get("CAMERA_CONFIG_FPS"))
# CAMERA_CONFIG_SECONDS = int(os.environ.get("CAMERA_CONFIG_SECONDS"))
# ######################################
# CAMERA_CONFIG_RETRY_CONNECTION = 5

# # Define metrics
# CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
# CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
# SENT_FRAME_BYTES = Summary('bytes_per_sent_frame', 'Number of bytes per sent frame')
# SENT_FRAMES = Counter('sent_frames', 'Total of sent frames')
# DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')

#DETECTED_CARS = Summary('detected_cars_per_frame', 'Number of detected cars per frame')
DETECTION_MODEL_PATH = './veiculo_novo_openvino_model'
DEVICE = os.environ.get("DEVICE_TYPE")
####### Source configuration #######
SERVICE_ID = os.environ.get("SERVICE_ID")
CAMERA_CONFIG_ID = os.environ.get("CAMERA_CONFIG_ID")
CAMERA_CONFIG_SOURCE = '../cam1_2.mp4'
#CAMERA_CONFIG_ROI_TOP_LEFT = eval(os.environ.get("CAMERA_CONFIG_ROI_TOP_LEFT")) #Formato "(500,500)"
#CAMERA_CONFIG_ROI_BOTTOM_RIGHT = eval(os.environ.get("CAMERA_CONFIG_ROI_BOTTOM_RIGHT"))
CAMERA_CONFIG_LOCAL = bool(os.environ.get("CAMERA_CONFIG_LOCAL"))
DEBUG = True
####### Teste em video local #######
CAMERA_CONFIG_FPS = 24
CAMERA_CONFIG_SECONDS = 90
######################################
CAMERA_CONFIG_RETRY_CONNECTION = 5

# Define metrics
CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
SENT_FRAME_BYTES = Summary('bytes_per_sent_frame', 'Number of bytes per sent frame')
SENT_FRAMES = Counter('sent_frames', 'Total of sent frames')
DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')


####### Define model config #######

model = YOLO(DETECTION_MODEL_PATH)

def process_video(camera):
    
    frame_id = 0

    loop = CAMERA_CONFIG_SECONDS*CAMERA_CONFIG_FPS if CAMERA_CONFIG_LOCAL else float('inf')
    
    port = 5555
    sender = imagezmq.ImageSender("tcp://*:{}".format(port), REQ_REP=False)
    


    while(frame_id < loop):
        
        sucess_capture, frame = camera.read()
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        all_vehicles = []
        results = model.predict(source=frame, conf=0.4, imgsz=640, verbose = False)
        for result in results:
            for detection in result.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                all_vehicles.append((frame[y1:y2, x1:x2].copy(), cls))

    # Se quiser mostrar o frame com a bounding box
        end_detection = datetime.now().timestamp()
        
        detection_time = end_detection - init_detection
        
        DETECTION_TIME.observe(detection_time)
        #DETECTED_CARS.observe(len(all_vehicles))

        if DEBUG: print("[DEBUG] Time:",end_detection,"| Frame ID:",frame_id,"| Detection Time:",detection_time,"| Number of detected cars:",len(all_vehicles))
        
        vehicle_id = 0

        for vehicle, c in all_vehicles:
            json_string = f'{{"frame_id": "{frame_id}", "vehicle_id": "{vehicle_id}", "vehicle_class": "{int(c)}"}}'
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
