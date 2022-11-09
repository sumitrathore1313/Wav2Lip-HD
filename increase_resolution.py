import sys

import glob

import cv2
import numpy as np
from tqdm import tqdm

import face_detection
from inference import face_detect, get_smoothened_boxes

import os
import subprocess
import platform

import cv2

PRED_VIDEO_PATH = "E:\\Wav2LipSKR\\temp\\fresult.avi"
PRED_IMAGES_PATH = "tecoGAN\\inputs"
SEP = '\\'
TECOGAN_PATH = "E:\\tecoGAN"
TECOGAN_INPUT_PATH = "E:\\Wav2LipSKR\\tecoGAN\\inputs"
TECOGAN_RESULT_PATH = "E:\\Wav2LipSKR\\tecoGAN\\results"
TECOGAN_LOG_PATH = "E:\\Wav2LipSKR\\tecoGAN\\log\\"
HD_VIDEO_PATH = "E:\\Wav2LipSKR\\temp\\result.avi"
AUDIO_PATH = "E:\\Wav2LipSKR\\inputs\\input1.mp3"

# read pred video and convert into images


# delete all images from folder
files = glob.glob('tecoGAN/inputs/*')
for f in files:
    os.remove(f)

files = glob.glob('tecoGAN/results/*')
for f in files:
    os.remove(f)

print('Convert Video frames')
pred_stream = cv2.VideoCapture(PRED_VIDEO_PATH)

pred_frames = []
pred_frame_count = 1000
while 1:
    still_reading, frame = pred_stream.read()
    if not still_reading:
        pred_stream.release()
        break

    cv2.imwrite(PRED_IMAGES_PATH + SEP + str(pred_frame_count) + '.png', np.array(frame))
    pred_frame_count += 1
    pred_frames.append(frame)

# make prediction using TecoGAN
os.chdir('E:\\tecoGAN')
print('Run tecoGAN')



def preexec():  # Don't forward signals.
    os.setpgrp()


def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn=preexec)


# run these test cases one by one:
cmd1 = ["C:\\ProgramData\\Anaconda3\\envs\\tecoGAN\\python.exe", "E:\\tecoGAN\\main.py",
        "--cudaID", "0",  # set the cudaID here to use only one GPU
        "--output_dir", TECOGAN_RESULT_PATH,  # Set the place to put the results.
        "--summary_dir", TECOGAN_LOG_PATH,  # Set the place to put the log.
        "--mode", "inference",
        "--input_dir_LR", TECOGAN_INPUT_PATH,  # the LR directory
        # "--input_dir_HR", os.path.join("./HR/", testpre[nn]),  # the HR directory
        # one of (input_dir_HR,input_dir_LR) should be given
        "--output_pre", TECOGAN_RESULT_PATH,  # the subfolder to save current scene, optional
        "--num_resblock", "16",  # our model has 16 residual blocks,
        # the pre-trained FRVSR and TecoGAN mini have 10 residual blocks
        "--checkpoint", './model/TecoGAN',  # the path of the trained model,
        "--output_ext", "png"  # png is more accurate, jpg is smaller
        ]
mycall(cmd1).communicate()

# create a Actual Video
print('Create HD Video')
os.chdir('E:\Wav2LipSKR')

video_stream = cv2.VideoCapture(HD_VIDEO_PATH)
fps = video_stream.get(cv2.CAP_PROP_FPS)

frames = []
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break
    frames.append(frame)

frame_h, frame_w = frames[0].shape[:-1]


detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False, device='cuda')

batch_size = 8

while 1:
    coordinates_preds = []
    try:
        for i in tqdm(range(0, len(frames), batch_size)):
            coordinates_preds.extend(detector.get_detections_for_batch(np.array(frames[i:i + batch_size])))
    except RuntimeError:
        if batch_size == 1:
            raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
        batch_size //= 2
        print('Recovering from OOM error; New batch size: {}'.format(batch_size))
        continue
    break

pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
coords = []
for rect, image in zip(coordinates_preds, frames):

    y1 = max(0, rect[1] - pady1)
    y2 = min(image.shape[0], rect[3] + pady2)
    x1 = max(0, rect[0] - padx1)
    x2 = min(image.shape[1], rect[2] + padx2)

    coords.append([x1, y1, x2, y2])


out = cv2.VideoWriter('tecoGAN/result.avi',
									cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
preds = []
for file in os.listdir(TECOGAN_RESULT_PATH):
    preds.append(np.array(cv2.imread(TECOGAN_RESULT_PATH + SEP + file)))

for p, f, c in zip(preds, frames, coords):
    x1, y1, x2, y2 = c

    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

    f[y1:y2, x1:x2] = p
    out.write(f)

command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(AUDIO_PATH, 'tecoGAN/result.avi', 'tecoGAN/result.mp4')
subprocess.call(command, shell=platform.system() != 'Windows')