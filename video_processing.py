# Importing Packages

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Setting Dimensions of Frame

frames = 15
Width = 256
Height = 256

# Loading Video Names

def load_video_names(path):
  videos = [] 
  labels = [] 
  for category in os.listdir(path): 
    if os.path.isdir(category): 
      for video in os.listdir(path+"/"+category): 
        videos.append(path+"/"+category+"/"+video) 
        labels.append(category)
  return np.array(videos), np.array(labels)

# Conversion of Frame Pixel

def preprocess(frame):
  frame = cv2.resize(frame, (Width, Height)) 
  frame = frame-127.5
  frame = frame/127.5
  return frame

# Loading Videos from Path

def load_video(video_path):
  video_frames = [] 
  cap = cv2.VideoCapture(video_path) 
  while True:
    ret, frame = cap.read()
    if ret == True:
      video_frames.append(preprocess(frame)) 
    else:
      break
  cap.release()
  video_frames = select_frames(video_frames) 
  if len(video_frames) != frames: 
    print('short_video ', video_path, len(video_frames))
    return 0, False

  return np.array(video_frames), True

# Choosing Desired No. of Frames

def select_frames(video_frames):
  selected_frames = []
  if len(video_frames) > frames:
    fn = len(video_frames)//frames 
    f_num = 0
    for f in video_frames:
      if len(selected_frames) < frames:
        if f_num % fn == 0:
          selected_frames.append(f)
      f_num += 1
  else:
    selected_frames = video_frames
  return selected_frames

# Load Batches of Videos

def create_dataset(videos, labels, indx):
  x = []
  y = []
  for video, label in zip(videos[indx], labels[indx]):
    xi, is_video = load_video(video)

    if is_video:
      x.append(xi)
      y.append(label)

  return np.array(x),np.array(y)
