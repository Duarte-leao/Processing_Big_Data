import cv2
import os

os.mkdir('images_giroslow')

video_name = "girosmallveryslow2.mp4" 
vidcap = cv2.VideoCapture(video_name)
success,image = vidcap.read()
count = 1
while success:
  cv2.imwrite("images_giroslow/%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1