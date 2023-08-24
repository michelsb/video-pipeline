import os
import cv2
import imagezmq
import json
from ultralytics import YOLO
import random
from torchvision import transforms

#from torchvision import transforms
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter
from util.preprocess import preprocess_image, image_to_tensor
from util.postprocess import postprocess
from openvino.runtime import Core
import torch
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
device = torch.device('cpu')

def load_dnn_model2():
    model=torch.load('./filter2 (1).pth')
    model = model.to(device)
    model.eval()
    return model
model = YOLO(DETECTION_MODEL_PATH)
if __name__ == "__main__":    

    start_http_server(8001)    
    
    port = 5555
    port_2 = 5556 # fiz isso não estou usando o docker
    image_hub = imagezmq.ImageHub("tcp://{}:{}".format(PREVIOUS_MODULE, port), REQ_REP=False)
    sender = imagezmq.ImageSender("tcp://*:{}".format(port_2), REQ_REP=False)
    filter_plate = load_dnn_model2()
    print(filter_plate)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])

    allFramesBytes = []
    boundBoxesH = []
    boundBoxesW = []

    all_detection_time = []
    all_filter_time = []
    filterNotDiscart = 0


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
                classe = cls.item()
                all_plates.append((frame[y1:y2, x1:x2].copy(), classe))

        end_detection = datetime.now().timestamp()

        detection_time = end_detection - init_detection
        
        #all_detection_time.append(detection_time)
        DETECTION_TIME.observe(detection_time)
        #DETECTED_PLATES.observe(len(all_plates))

        filterNotDiscart = 0
        all_detection_time.append(detection_time)

        for plate,c in all_plates:      
            plate_id = 0

            json_object["plate_id"] = plate_id
            plate.nbytes
            allFramesBytes.append(plate.nbytes)
            plate.shape
            h,w,_ = plate.shape
            boundBoxesH.append(h)
            boundBoxesW.append(w)
            json_object["plate_class"] = int(c)

            plate_tensor = transform( cv2.resize(plate, (224,224)) )
            batch = plate_tensor.unsqueeze( 0)
            batch = batch.to(device)

            init_filter = datetime.now().timestamp()
            
            out = filter_plate(batch)
            _, filter_class = torch.max(out, 1)
            print(filter_class)
            end_filter = datetime.now().timestamp()

            filter_time = end_filter - init_filter

            all_filter_time.append(filter_time)

            if filter_class in [0]:
                
                    filter_class = int(filter_class)
                    filterNotDiscart += 1
                    print(json_object)
                    json_object["filter_class"] = int(filter_class)
                    new_json_string = json.dumps(json_object)
                    #print(new_json_string)
                    sender.send_image(new_json_string, plate)
            
            plate_id +=1

        #image_hub.send_reply(b'OK')
        try:
                print("plate_id")
                print("\n")
                print(f"Tempo de inferência médio da detecção de placas (Inferência e Recorte): {sum(all_detection_time)/len(all_detection_time)}")
                print(f"Tamanho médio das placas (bytes): {sum(allFramesBytes)/len(allFramesBytes)}")
                print(f"Total de placas detectados: {len(boundBoxesH)}")
                print(f"Tamanho Médio dos Bound Boxes das placas: (h={sum(boundBoxesH)/len(boundBoxesH)},w={sum(boundBoxesW)/len(boundBoxesW)})")
                print(f"Tempo de inferência médio da filtragem de placas (Inferência): {sum(all_filter_time)/len(all_filter_time)}")
                print(f"Total de placas não descratadas: {filterNotDiscart}")
                print(f"Total de placas descratadas: {len(boundBoxesH)-filterNotDiscart}")
        except:
                pass            
    sender.zmq_socket.close()
