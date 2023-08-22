import os
import cv2
import imagezmq
import json
import time
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
DETECTION_TIME = Summary('detection_time_seconds_per_frame', 'Inference time (Inference and Clipping) per frame')

directory = '/data'

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
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, nc=36)
    return detections

def translate(results, class_names):
    boxes = results["det"]    
    if not len(boxes): return ""
    #boxes = results["det"].copy()
    #classes = boxes[boxes[:, 0].argsort()][:,-1]
    classes = boxes[boxes[:, 0].argsort()][:,-1]
    return "".join(list(map(lambda x: class_names[int(x)], classes)))

if __name__ == "__main__":

    start_http_server(8002)

    class_names = []
    
    with open("ocr-net.names", "r") as f:
	    class_names = [cname.strip() for cname in f.readlines()]
    
    #image_hub = imagezmq.ImageHub(open_port=LOCAL_ADD, REQ_REP=False)#, REQ_REP=False)    
    #sender = imagezmq.ImageSender(connect_to=ADDRESS_NEXT_MODULE)#, REQ_REP=False)
    port = 5555
    image_hub = imagezmq.ImageHub("tcp://{}:{}".format(PREVIOUS_MODULE, port), REQ_REP=False)

    ocr = load_dnn_model()
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    number_plates = 0
    #all_ocr_time = []

    # Change the current directory 
    # to specified directory 
    os.chdir(directory)

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

        frame = cv2.resize(frame, (200,80), interpolation = cv2.INTER_AREA)
        
        detections = detect(frame.copy(), ocr)[0]
        
        text_plate = translate(detections, class_names)

        json_object["plate_text"] = text_plate
        
        end_ocr = datetime.now().timestamp()
        
        ocr_time = end_ocr - init_ocr
        
        #all_ocr_time.append(ocr_time)
        DETECTION_TIME.observe(ocr_time)
        if DEBUG: print("[DEBUG] Time:",end_ocr,"| Frame ID:",json_object["frame_id"],"| Vehicle ID:", json_object["vehicle_id"],"| Plate ID:",json_object["plate_id"],"| Plate Class: ",json_object["plate_class"],"| OCR Time:",ocr_time,"| Plate Text:", text_plate)
    
