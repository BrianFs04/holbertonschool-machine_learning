#!/usr/bin/env python3
"""Yolo (You only look once)"""
import tensorflow.keras as K
import tensorflow as tf
import numpy as np


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor method
            model: the Darknet Keras model
            class_names: a list of the class names for the model
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Returns sigmoid function"""
        return(1/(1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """Returns the processed boundary boxes for each output"""
        boxes, box_confidences, box_class_probs = [], [], []
        img_h, img_w = image_size
        for i in range(len(outputs)):
            # input sizes
            input_w = self.model.input_shape[1]
            input_h = self.model.input_shape[2]

            # Grid height, grid width and anchors boxes
            grid_h = outputs[i].shape[0]
            grid_w = outputs[i].shape[1]
            anchor_boxes = outputs[i].shape[2]

            # Predicted coordinates
            tx = outputs[i][..., 0]
            ty = outputs[i][..., 1]
            tw = outputs[i][..., 2]
            th = outputs[i][..., 3]

            # corner
            c = np.zeros((grid_h, grid_w, anchor_boxes))
            # indexes and top-left corner
            idx_y = np.arange(grid_h)
            idx_y = idx_y.reshape(grid_h, 1, 1)
            idx_x = np.arange(grid_w)
            idx_x = idx_x.reshape(1, grid_w, 1)
            cx = c + idx_x
            cy = c + idx_y

            # Anchors width and height
            # [116 156 373] [ 90 198 326]
            # [30 62 59] [ 61 45 119]
            # [10 16 33] [13 30 23]
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # Bounding box prediction
            bx = self.sigmoid(tx) + cx
            by = self.sigmoid(ty) + cy
            bw = pw * np.exp(tw)
            bh = ph * np.exp(th)

            # normalize bx and by values to the grid
            bx = bx / grid_w
            by = by / grid_h

            # normalize bw and bh values to the input sizes
            bw = bw / input_w
            bh = bh / input_h

            # get the corner coordinates
            bx1 = bx - bw / 2
            by1 = by - bh / 2
            bx2 = bx + bw / 2
            by2 = by + bh / 2

            # to image size scale
            outputs[i][..., 0] = bx1 * img_w
            outputs[i][..., 1] = by1 * img_h
            outputs[i][..., 2] = bx2 * img_w
            outputs[i][..., 3] = by2 * img_h

            # filtered bounding boxes
            boxes.append(outputs[i][..., 0:4])
            # objectiveness score between 0 and 1
            box_confidences.append(self.sigmoid(outputs[i][..., 4:5]))
            # probability of classes
            box_class_probs.append(self.sigmoid(outputs[i][..., 5:]))
        return(boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """filter out boxes with low object score"""
        scores = []

        for i in range(len(boxes)):
            # Computing box scores
            scores.append(box_confidences[i] * box_class_probs[i])

        # filtering boxes
        filter_boxes = [box.reshape(-1, 4) for box in boxes]
        filter_boxes = np.concatenate(filter_boxes)

        # Finding the index of the class with maximum box score
        classes = [np.argmax(box, -1) for box in scores]
        classes = [box.reshape(-1) for box in classes]
        classes = np.concatenate(classes)

        # Getting the corresponding box score
        class_scores = [np.max(box, -1) for box in scores]
        class_scores = [box.reshape(-1) for box in class_scores]
        class_scores = np.concatenate(class_scores)

        filtering_mask = np.where(class_scores >= self.class_t)
        # Applying the mask to boxes, classes and scores
        filtered_boxes = filter_boxes[filtering_mask]
        box_classes = classes[filtering_mask]
        box_scores = class_scores[filtering_mask]

        return(filtered_boxes, box_classes, box_scores)

    def iou(self, filtered_boxes, scores):
        """Returns the intersection over union result"""
        # grab the coordinates of the bounding boxes
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = scores.argsort()[::-1]

        # initialize the list of picked indexes
        pick = []
        # keep looping while some indexes still remain in the indexes
        # list
        while idxs.size > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            i = idxs[0]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            inter = w * h
            union = area[i] + area[idxs[1:]] - inter
            overlap = inter / union

            # delete all indexes from the index list that have
            ind = np.where(overlap <= self.nms_t)[0]
            idxs = idxs[ind + 1]

        # return only the bounding boxes that were picked using the
        # integer data type
        return pick

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """non max suppression"""
        box_predictions = []
        predicted_box_classes, predicted_box_score = [], []
        u_classes = np.unique(box_classes)
        for cls in u_classes:
            idx = np.where(box_classes == cls)

            filters = filtered_boxes[idx]
            scores = box_scores[idx]
            classes = box_classes[idx]

            pick = self.iou(filters, scores)

            filters1 = filters[pick]
            scores1 = scores[pick]
            classes1 = classes[pick]

            box_predictions.append(filters1)
            predicted_box_classes.append(classes1)
            predicted_box_score.append(scores1)
        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(predicted_box_classes, axis=0)
        predicted_box_score = np.concatenate(predicted_box_score, axis=0)

        return (box_predictions, predicted_box_classes, predicted_box_score)
