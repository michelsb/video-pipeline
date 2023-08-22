import numpy as np
import time


def postprocess(
    pred_boxes, input_hw,
    orig_img, min_conf_threshold = 0.25,
    nms_iou_threshold = 0.7, agnosting_nms = False,
    max_detections = 300,
    nc=0
):
    
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    preds = non_max_suppression(
        pred_boxes,
        min_conf_threshold,
        nms_iou_threshold,
        nc=nc,
        **nms_kwargs
    )
    results = []

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": []})
            continue
        else:
            pred[:, :4] = scale_boxes(input_hw, pred[:, :4], shape).round()
        results.append({"det": pred[:, :6]})
    return results


def non_max_suppression(
    prediction,
    conf_thres = 0.7,
    iou_thres = 0.45,
    classes = None,
    agnostic = False,
    multi_label = False,
    labels = (),
    max_wh = 4096,
    max_det = 300,
    max_nms = 30000 ,
    nc=0

):
    
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    time_limit = 10.0
    redundant = True 
    multi_label &= nc > 1
    merge = False

    t = time.time()
    output = [np.zeros((0, 6))
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if labels and len(labels[xi]):
            lxi = labels[xi]
            v = np.zeros((len(lxi), nc + 5))
            v[:, :4] = lxi[:, 1:5]
            v[:, 4] = 1.0 
            v[range(len(lxi)), lxi[:, 0].long() + 5] = 1.0
            x = np.concatenate((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        conf, j = x[:, 5:].max(1, keepdims=True), x[:, 5:].argmax(1, keepdims=True)
        x = np.concatenate((box, conf, j.astype(float)), 1)[conf.reshape(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms_python(boxes, scores, iou_thres)
        if merge and (1 < n < 3E3):
            iou = box_iou(boxes[i], boxes) > iou_thres
            weights = iou * scores[None]
            x[i, :4] = np.dot(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return output


def nms_python(bboxes, psocres, threshold):
    
      bboxes = bboxes.astype('float')
      x_min = bboxes[:,0]
      y_min = bboxes[:,1]
      x_max = bboxes[:,2]
      y_max = bboxes[:,3]
      
      sorted_idx = psocres.argsort()[::-1]
      bbox_areas = (x_max-x_min+1)*(y_max-y_min+1)
      
      filtered = []
      while len(sorted_idx) > 0:
          rbbox_i = sorted_idx[0]
          filtered.append(rbbox_i)
          
          overlap_xmins = np.maximum(x_min[rbbox_i],x_min[sorted_idx[1:]])
          overlap_ymins = np.maximum(y_min[rbbox_i],y_min[sorted_idx[1:]])
          overlap_xmaxs = np.minimum(x_max[rbbox_i],x_max[sorted_idx[1:]])
          overlap_ymaxs = np.minimum(y_max[rbbox_i],y_max[sorted_idx[1:]])
          
          overlap_widths = np.maximum(0,(overlap_xmaxs-overlap_xmins+1))
          overlap_heights = np.maximum(0,(overlap_ymaxs-overlap_ymins+1))
          overlap_areas = overlap_widths*overlap_heights
          
          ious = overlap_areas/(bbox_areas[rbbox_i]+bbox_areas[sorted_idx[1:]]-overlap_areas)
          
          delete_idx = np.where(ious > threshold)[0]+1
          delete_idx = np.concatenate(([0],delete_idx))
          
          sorted_idx = np.delete(sorted_idx,delete_idx)
          
      
      return filtered


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (np.min(a2, b2) - np.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def clip_boxes(boxes, shape):
    
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    
      if ratio_pad is None:
          gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
          pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
              (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
      else:
          gain = ratio_pad[0][0]
          pad = ratio_pad[1]

      boxes[..., [0, 2]] -= pad[0]
      boxes[..., [1, 3]] -= pad[1]
      boxes[..., :4] /= gain
      clip_boxes(boxes, img0_shape)
      return boxes