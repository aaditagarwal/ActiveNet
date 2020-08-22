import os
import argparse
from pathlib import Path
import datetime

import cv2
import numpy as np
import torch
import json
import time

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from utils import normalize, pad_width
from create_encoding import generate_angles_between_joints

import sandesh
import joblib
import warnings
warnings.filterwarnings("ignore")

# Loading the classifier
clf = joblib.load("classifiers/1-acc_mask.pkl")
# Loading the scaler
scaler = joblib.load("scalers/stdscaler_masked.pkl")

class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        print("file read:", self.file_names[self.idx])
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
    stages_output = net(tensor_img)
    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    
    return heatmaps, pafs, scale, pad

def run_demo(net, image_provider, height_size, cpu, track, smooth, json_out):
    net = net.eval()
    if not cpu:
        net = net.cuda()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    # dict_to_dump = {}

    # Extracting the webhook from environment variable
    webhook = os.environ['ACTIVENET_WEBHOOK']
    
    # Original delay in sending alert when below 25% label found
    delay = 0

    for img in image_provider:
        now = time.time()

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        current_poses = []
        pred = None
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            
            # Used to dump training data
            # print(pose_keypoints.shape)
            # dict_to_dump[filename[8:]] = {}
            # dict_to_dump[filename[8:]]['body_keypoints'] = pose_keypoints.tolist()
 
            # Generating encoding from keypoints
            encoding = generate_angles_between_joints(pose_keypoints)
            # dict_to_dump[filename[8:]]['encoding'] = encoding
            
            values = encoding
            # Masking
            encoding = np.where(np.isnan(values), np.ma.array(values, mask=np.isnan(values)).mean(axis=0), values)
            
            # Preparing for input to the model
            input_enc = np.expand_dims(encoding, 0)
            
            # Scaling the encoding
            std_input = scaler.transform(input_enc)
            
            # Getting prediction from the model
            pred = clf.predict(std_input)
            current_poses.append(pose)
            
            # Currently use just a single person detection
            break

        # Tracking
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)

        
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))

        img = cv2.addWeighted(orig_img, 0.5, img, 0.5, 0)


        # img = cv2.resize(img, (640,480))

        # out is the string to display over OpenCV Image
        if pred is None:
            out = "No Person Found."
            cv2.putText(img,out,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        
        elif pred[0] == 0:
            out = "Below 25%"

            # Put the alert in a dictionary
            alert = {
                "ALERT" : "Attentiveness is below 25%!",
                "TIMESTAMP" : str(datetime.datetime.now()),
            }
            # If more than 15 contiguous frames are classified as Below 25%, trigger the alert
            if delay > 10:
                print(alert)
                delay = 0
                
                # Send the message
                sandesh.send(alert, use_raw=True, webhook=webhook)

            cv2.putText(img,out,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

            delay += 1

        elif pred[0] == 1:
            out = "Between 25 & 50%"
            delay = 0
            cv2.putText(img,out,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)

        elif pred[0] == 2:
            out = "Between 50 & 75%"
            delay = 0
            cv2.putText(img,out,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)

        elif pred[0] == 3:
            out = "Above 75%"
            delay = 0
            cv2.putText(img,out,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)


        then = time.time()
        # print(f"Frame processed in {then-now} seconds.")


        cv2.imshow("Output", img)
        key = cv2.waitKey(1)
        if key == 27:  # esc
            return
    # Used to dump training data
    # file_json = open(json_out, "w+")  #json_out is full file path
    # file_json.write(json.dumps(dump_encoding, indent=1))
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Activeness-Detection.''')
    parser.add_argument('--source', type=str, default='0', help='path to video file or camera id')
    parser.add_argument('--json_out', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, help='path to the checkpoint', default="weights/checkpoint_iter_370000.pth")
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu', default=False)
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')

    args = parser.parse_args()

    if args.source == '':
        raise ValueError('--source has to be provided')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    frame_provider = VideoReader(args.source)
    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, args.json_out)