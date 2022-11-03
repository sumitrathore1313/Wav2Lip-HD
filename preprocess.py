"""
preprocess the lrs2 dataset to lrs2_preprocessed folder.
It will be used in training the wav2lip-HD model.
It is change for using only one GPU.
"""
import argparse
import os
import subprocess
import sys
import traceback
from glob import glob
from os import path

import cv2
import numpy as np
from tqdm import tqdm

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument('--device', help='Use cuda or cpu', default="cuda")
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", default="data_root/main")
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", default="lrs2_preprocessed")
parser.add_argument("--ffmpeg_path", help="Path of FFMPEG", default="C:\\ffmpeg\\bin\\ffmpeg")
parser.add_argument("--verbose", help="Path of FFMPEG", default=False)
parser.add_argument("--os", help="Window / Linux / Mac", default="window")

args = parser.parse_args()

seperator = '\\' if args.os == 'window' else '/'

# Get the face detector
face_detector = 'sfd'
face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                  globals(), locals(), [face_detector], 0)
face_detector_kwargs = {}
face_detector = face_detector_module.FaceDetector(
    device=args.device,
    verbose=args.verbose,
    **face_detector_kwargs
)

template = args.ffmpeg_path + ' -loglevel panic -y -i {} -strict -2 {}'


def process_video_file(file):
    # get the video stream for getting frames
    video_stream = cv2.VideoCapture(file)

    # save each frame of videos in frames list
    frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    # get the name of video and directory
    name = os.path.basename(file).split('.')[0]
    dirname = file.split(seperator)[-2]

    # create the parent and video dir in preprocessed folder
    full_dir = path.join(args.preprocessed_root, dirname, name)
    os.makedirs(full_dir, exist_ok=True)

    # create batches which num of frame specified in batch size
    batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]

    i = -1
    for batch in batches:
        # get the predictions for batch
        for index, image in enumerate(batch):
            i += 1
            prediction = face_detector.detect_from_image(image.copy())
            if len(prediction) == 0:
                continue
            d = prediction[0]
            d = np.clip(d, 0, None)
            x1, y1, x2, y2 = map(int, d[:-1])
            cv2.imwrite(path.join(full_dir, '{}.jpg'.format(i)), batch[index][y1:y2, x1:x2])


def process_audio_file(file):
    # get the name of video and directory
    name = os.path.basename(file).split('.')[0]
    dirname = file.split(seperator)[-2]

    # create the parent and video dir in preprocessed folder
    full_dir = path.join(args.preprocessed_root, dirname, name)

    wav_path = path.join(full_dir, 'audio.wav')

    # use ffmpeg to generate audio from video
    command = template.format(file, wav_path)
    subprocess.call(command, shell=True)


def main():
    print('Started processing for {}'.format(args.data_root))

    # get all the mp4 files from main folder of lrs2
    filelist = glob(path.join(args.data_root, '*/*.mp4'))

    print('No of Video found - ' + str(len(filelist)))
    print('Dumping images...')

    # process the video into images
    for file in tqdm(filelist):
        try:
            process_video_file(file)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()

    print('Dumping audios...')

    # process the video into audio
    for file in tqdm(filelist):
        try:
            process_audio_file(file)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main()
