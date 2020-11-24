#!/usr/bin/env python3
from threading import Thread
import numpy as np

import rospy  # Python library for ROS
from sensor_msgs.msg import Image  # Image is the message type
from std_msgs.msg import String
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
import argparse
import torch
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def callback(data):
    # Used to convert between ROS and OpenCV images
    br = CvBridge()

    # Output debugging information to the terminal
    rospy.loginfo("receiving video frame")

    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)
    img0 = current_frame
    img = detector.get_img_for_model(img0)
    current_frame, obj_to_pub = detector.analyze(img, img0)
    # Display image
    for s in obj_to_pub:
        print(s)
        pub.publish(s)


    cv2.imshow("camera", current_frame)

    cv2.waitKey(1)


def receive_message_pub_data():
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.
    rospy.init_node('video_sub_boxes_send_py', anonymous=True)

    # Node is subscribing to the video_frames topic
    rospy.Subscriber('video_frames', Image, callback)
    global pub
    pub = rospy.Publisher('data_from_images', String, queue_size=10)
    rospy.Rate=10
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    # Close down the video stream when done
    cv2.destroyAllWindows()




class Detector:
    def __init__(self, weights, imgsz):
            self.weights = weights


            # Initialize
            self.device = select_device(opt.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
            self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
            if self.half:
                self.model.half()  # to FP16

            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

            # Run inference
            img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
            self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def get_img_for_model(self, img0):
        img = [self.letterbox(img0, new_shape=self.imgsz)[0]]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        return img

    def analyze(self, img, im0s):


                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = self.model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    print(pred)
                    # if webcam:  # batch_size >= 1
                    #     p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                    # else:
                    #     p, s, im0 = Path(path), '', im0s
                    im0 = im0s
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    objs_to_publish = []
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        # for c in det[:, -1].unique():
                        #     n = (det[:, -1] == c).sum()  # detections per class
                        #     s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            objs_to_publish.append(f"{label} {xywh}")
                            # plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)


                    else:
                        print("Found nothing")
                    return im0, objs_to_publish
                    # Stream results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    global detector
    detector = Detector(opt.weights, opt.img_size)
    receive_message_pub_data()
