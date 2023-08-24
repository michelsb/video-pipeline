import os
import sys
import traceback
import cv2
import imagezmq
import json
import time
import torch
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter
from util.preprocess import preprocess_image, image_to_tensor
from util.postprocess import postprocess
from openvino.runtime import Core

####### Device configuration #######
# DETECTION_MODEL_PATH = os.environ.get("DETECTION_MODEL_PATH")
# DEVICE = os.environ.get("DEVICE_TYPE")
# ####### Source configuration #######
# PREVIOUS_MODULE = os.environ.get("PREVIOUS_MODULE")
# DEBUG = bool(os.environ.get("DEBUG"))

# # Define metrics
# CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
# CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
# DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')
DETECTION_MODEL_PATH = './OCR_novo.onnx'
DEVICE = os.environ.get("DEVICE_TYPE")
####### Source configuration #######
PREVIOUS_MODULE = '127.0.0.1'
DEBUG = True

# Define metrics
CAPTURED_FRAME_BYTES = Summary('bytes_per_captured_frame', 'Number of bytes per captured frame')
CAPTURED_FRAMES = Counter('captured_frames', 'Total of captured frames')
DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')
directory = '/data'
device = torch.device('cpu')
OCR_model = torch.hub.load('ultralytics/yolov5', 'custom', path=DETECTION_MODEL_PATH, device=device)
# def load_dnn_model():
#     core = Core()
#     det_ov_model = core.read_model(DETECTION_MODEL_PATH)
#     if DEVICE != "CPU":
#         det_ov_model.reshape({0: [1, 3, 640, 640]})
#     det_compiled_model = core.compile_model(det_ov_model, DEVICE)
#     return det_compiled_model

# def detect(image, model):
#     preprocessed_image = preprocess_image(image)
#     input_tensor = image_to_tensor(preprocessed_image)
#     result = model(input_tensor)
#     boxes = result[model.output(0)]
#     input_hw = input_tensor.shape[2:]
#     detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, nc=36)
#     return detections

# def translate(results, class_names):
#     boxes = results["det"]    
#     if not len(boxes): return ""
#     #boxes = results["det"].copy()
#     #classes = boxes[boxes[:, 0].argsort()][:,-1]
#     classes = boxes[boxes[:, 0].argsort()][:,-1]
#     return "".join(list(map(lambda x: class_names[int(x)], classes)))

if __name__ == "__main__":

    start_http_server(8002)

    class_names = []
    
    with open("ocr-net.names", "r") as f:
      class_names = [cname.strip() for cname in f.readlines()]
    
    #image_hub = imagezmq.ImageHub(open_port=LOCAL_ADD, REQ_REP=False)#, REQ_REP=False)    
    #sender = imagezmq.ImageSender(connect_to=ADDRESS_NEXT_MODULE)#, REQ_REP=False)
    port = 5556
    image_hub = imagezmq.ImageHub("tcp://{}:{}".format(PREVIOUS_MODULE, port), REQ_REP=False)
    print('a')
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    number_plates = 0
    #all_ocr_time = []

    # Change the current directory 
    # to specified directory 
    #os.chdir(directory)

    while True:
        json_string, frame = image_hub.recv_image()

        # Filename
        #filename = "savedImage"+str(number_plates)+".jpg"
        
        # Using cv2.imwrite() method
        # Saving the image
        #cv2.imwrite(filename, frame)
        #print(json_string)
        CAPTURED_FRAME_BYTES.observe(frame.nbytes)
        CAPTURED_FRAMES.inc()

        number_plates += 1
        
        json_object = json.loads(json_string)
        
        init_ocr = datetime.now().timestamp()

        frame = cv2.resize(frame, (640,640), interpolation = cv2.INTER_AREA) # não tenho nenhum modelo de ocr v8 e tive que fazer essa atrocidade, Deus me perdoe

        def ocr_detect(frame):
            global OCR_model

            text_plate = None

            try:

                rets = []
                text_plate = ""
                plate_list = []

                
                res = OCR_model(frame)
                for i in res.xyxy[0]:
                        x1 = int(i[0])
                        y1 = int(i[1])
                        x2 = int(i[2])
                        y2 = int(i[3])
                        rets.append([[x1,y1,x2-x1,y2-y1], int(i[5])])



                for boxL, classId in rets:

                    if boxL[2]*boxL[3] > 700:
                        continue
                    stop = False
                    for b in plate_list:
                        if abs(boxL[0]-b[0]) < 5:
                            stop = True
                            break
                    if stop:
                        continue
                    char = class_names[classId]
                    cv2.rectangle(frame, boxL, (255,0,0), 1)
                    cv2.putText(frame, char, (boxL[0]+2,boxL[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    if len(plate_list)==0:
                        plate_list.append((boxL[0],char))
                    else:
                        tamList = len(plate_list)
                        inserido = False
                        for i in range(tamList):
                            if plate_list[i][0] > boxL[0]:
                                plate_list.insert(i,(boxL[0],char))
                                inserido = True
                                break
                        if not inserido:
                            plate_list.append((boxL[0],char))

                plate_list = [x2 for x1,x2 in plate_list]
                text_plate = "".join(plate_list)
                    
            except:
                traceback.print_exc()
                sys.exit(1)

            return text_plate
        text_plate = ocr_detect(frame)
        print(text_plate)

        json_object["plate_text"] = text_plate
        
        end_ocr = datetime.now().timestamp()
        
        ocr_time = end_ocr - init_ocr
        
        #all_ocr_time.append(ocr_time)
        DETECTION_TIME.observe(ocr_time)
        if DEBUG: print("[DEBUG] Time:",end_ocr,"| Frame ID:",json_object["frame_id"],"| Vehicle ID:", json_object["vehicle_id"],"| Plate ID:",json_object["plate_id"],"| Plate Class: ",json_object["plate_class"],"| OCR Time:",ocr_time,"| Plate Text:", text_plate)
    
