import cv2
import os
import uuid
  
# Read the video from specified path
cam = cv2.VideoCapture("input_videos/3.mp4")
  
try:
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')  
except OSError:
    print ('Error: Creating directory of data')
    
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        name = str(uuid.uuid4())
        name = name.replace('-', '')
        name = os.path.join('output_images', f"{name}.jpg")
        print ('Creating...' + name)
  
        # writing the extracted images
        if  currentframe %30 == 0:
            cv2.imwrite(name, frame)
  
        # increasing counter so that it will
        # show how many frames are created
       
    else:
        break
        
    currentframe += 1

cam.release()
cv2.destroyAllWindows()