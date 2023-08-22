import os
import cv2
import imagezmq
import json
#from torchvision import transforms
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter
from util.preprocess import preprocess_image, image_to_tensor
from util.postprocess import postprocess
from openvino.runtime import Core

####### Device configuration #######
DETECTION_MODEL_PATH = os.environ.get("DETECTION_MODEL_PATH")
DEVICE = os.environ.get("DEVICE_TYPE")
####### Source configuration #######
PREVIOUS_MODULE = os.environ.get("PREVIOUS_MODULE")
DEBUG = bool(os.environ.get("DEBUG"))

# Define metrics
CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
SENT_FRAME_BYTES = Summary('bytes_per_sent_frame', 'Number of bytes per sent frame')
SENT_FRAMES = Counter('sent_frames', 'Total of sent frames')
DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')

#DETECTED_FRAME_BYTES = Summary('bytes_per_detected_frame', 'Number of bytes per detected plate')
#DETECTED_PLATES = Summary('detected_plates_per_frame', 'Number of detected plates per frame')
#DROPPED_PLATES = Summary('dropped_plates', 'Number of dropped plates per frame')

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
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, nc=7)
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

if __name__ == "__main__":    

    start_http_server(8001)    
    
    port = 5555
    image_hub = imagezmq.ImageHub("tcp://{}:{}".format(PREVIOUS_MODULE, port), REQ_REP=False)
    sender = imagezmq.ImageSender("tcp://*:{}".format(port), REQ_REP=False)

    plate_detection = load_dnn_model()

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
        
        detections = detect(frame.copy(), plate_detection)[0]
        
        all_plates = cut(detections, frame)

        end_detection = datetime.now().timestamp()

        detection_time = end_detection - init_detection
        
        #all_detection_time.append(detection_time)
        DETECTION_TIME.observe(detection_time)
        #DETECTED_PLATES.observe(len(all_plates))

        filterNotDiscart = 0

        for plate,c in all_plates:      
            plate_id = 0

            json_object["plate_id"] = plate_id

            #DETECTED_FRAME_BYTES.observe(plate.nbytes)            

            json_object["plate_class"] = c
            
            if DEBUG: print("[DEBUG] Time:",datetime.now().timestamp(),"| Frame ID:",json_object["frame_id"],"| Vehicle ID:", json_object["vehicle_id"],"| Detection Time:",detection_time,"| Number of detected plates: ",len(all_plates),"| Plate ID:",plate_id,"| Plate Class: ",c)

            if c in [0,1,2,4,5]:
                filterNotDiscart += 1
                json_object["filter_class"] = c in [0,4]
                new_json_string = json.dumps(json_object)
                #print(new_json_string)
                SENT_FRAME_BYTES.observe(plate.nbytes)
                SENT_FRAMES.inc()
                sender.send_image(new_json_string, plate)           
            
            plate_id +=1

        #DROPPED_PLATES.observe(len(all_plates)-filterNotDiscart)

    sender.zmq_socket.close()
