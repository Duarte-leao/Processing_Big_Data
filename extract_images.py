import cv2
import os

os.mkdir('images')

video_name = "bestofgiro2022.mp4" # or any other extension like .avi etc
vidcap = cv2.VideoCapture(video_name)
success,image = vidcap.read()
count = 1
while success:
  cv2.imwrite("images/%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1