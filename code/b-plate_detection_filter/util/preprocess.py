import cv2
import numpy as np


def preprocess_image(img0):

    img = letterbox(img0)[0]

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def image_to_tensor(image):

    input_tensor = image.astype(np.float32)
    input_tensor /= 255.0

    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def letterbox(img, new_shape = (640, 640), color = (114, 114, 114),
              auto = False, scale_fill = False, scaleup = False, stride = 32):
    
      shape = img.shape[:2]
      if isinstance(new_shape, int):
          new_shape = (new_shape, new_shape)

      r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
      if not scaleup:
          r = min(r, 1.0)

      ratio = r, r
      new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
      dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
      if auto:
          dw, dh = np.mod(dw, stride), np.mod(dh, stride)
      elif scale_fill:
          dw, dh = 0.0, 0.0
          new_unpad = (new_shape[1], new_shape[0])
          ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

      dw /= 2
      dh /= 2

      if shape[::-1] != new_unpad:
          img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
      top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
      left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
      img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
      return img, ratio, (dw, dh)
