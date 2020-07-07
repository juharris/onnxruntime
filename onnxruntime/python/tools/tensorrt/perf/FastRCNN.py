import os
import sys
import numpy as np
import onnx
import onnxruntime
import subprocess
from PIL import Image
from BaseModel import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class FastRCNN(BaseModel):
    def __init__(self, model_name='Fast R-CNN'): 
        BaseModel.__init__(self, model_name)
        self.inputs_ = []
        self.ref_outputs_ = []
        self.outputs_ = []

        if not os.path.exists("FasterRCNN-10"):
            subprocess.run("wget https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.tar.gz", shell=True, check=True)
            subprocess.run("tar zxf FasterRCNN-10.tar.gz FasterRCNN-10", shell=True, check=True)

        self.image_ = Image.open('dependencies/demo.jpg')
        self.preprocess()

        try: 
            self.session_ = onnxruntime.InferenceSession('FasterRCNN-10/faster_rcnn_R_50_FPN_1x.onnx')
        except:
            subprocess.run("python3 ../symbolic_shape_infer.py --input ./FasterRCNN-10/faster_rcnn_R_50_FPN_1x.onnx --output ./FasterRCNN-10/faster_rcnn_R_50_FPN_1x_new.onnx --auto_merge", shell=True, check=True)     
            self.session_ = onnxruntime.InferenceSession('FasterRCNN-10/faster_rcnn_R_50_FPN_1x_new.onnx')


    def preprocess(self):
        image = self.image_

        # Resize
        ratio = 800.0 / min(image.size[0], image.size[1])
        image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

        # Convert to BGR
        image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

        # HWC -> CHW
        image = np.transpose(image, [2, 0, 1])

        # Normalize
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(image.shape[0]):
            image[i, :, :] = image[i, :, :] - mean_vec[i]

        # Pad to be divisible of 32
        import math
        padded_h = int(math.ceil(image.shape[1] / 32) * 32)
        padded_w = int(math.ceil(image.shape[2] / 32) * 32)

        padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
        padded_image[:, :image.shape[1], :image.shape[2]] = image
        image = padded_image

        self.image_ = image

    def inference(self):
        session = self.session_
        img_data = self.image_

        boxes, labels, scores = session.run(None, {
            session.get_inputs()[0].name: img_data
        })

        self.boxes_ = boxes
        self.labels_ = labels
        self.scores_ = scores

    def postprocess(self):
        import matplotlib as mpl
        mpl.use('Agg')

        classes = [line.rstrip('\n') for line in open('dependencies/coco_classes.txt')]
        classes[0:5]

        image = Image.open('dependencies/demo.jpg')
        boxes = self.boxes_
        labels = self.labels_
        scores = self.scores_
        score_threshold=0.7

        # Resize boxes
        ratio = 800.0 / min(image.size[0], image.size[1])
        boxes = boxes / ratio

        fig = plt.figure(figsize=(12,9))
        ax = fig.add_subplot(1,1,1)
        ax.imshow(image)

        # Showing boxes with score > 0.7
        for box, label, score in zip(boxes, labels, scores):
            if score > score_threshold:
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
                ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
                ax.add_patch(rect)
        fig.savefig('result.png')


        '''
        pred = boxes
        import cv2
        viz = cv2.imread('dependencies/demo.jpg')
        for i in range(pred.shape[0]):
            # skip low confidence detection
            if scores[i] < score_threshold:
                continue
            cv2.rectangle(viz, (pred[i, 0], pred[i, 1]), (pred[i, 2], pred[i, 3]), (0, 255, 0), 2)
            print('%d %d %d %d %f' % (
                round(pred[i, 0]), round(pred[i, 1]),
                round(pred[i, 2] - pred[i, 0]), round(pred[i, 3] - pred[i, 1]),
                scores[i]))

        cv2.imwrite('result.jpg', viz)
        '''



