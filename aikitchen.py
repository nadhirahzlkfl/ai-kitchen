#%% import packages
from ultralytics import YOLO
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import os

model=YOLO('yolov8n.pt')
# data loading
data_path=os.path.join(os.getcwd(),'dataset','data.yaml')
# model training
result=model.train(data=data_path,epochs=50,imgsz=640)

# model prediction
filepath=os.path.join(os.getcwd(),'static','sample image.jpg')
result=model(source=filepath)
print(result)
img=result[0].plot()
img=cv2.cvtColor(img,code=cv2.COLOR_BGR2RGB) # convert to RGB
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()

# prediction with webcam stream
result=model(source=0,show=True,stream=True)
print(result)

cam=cv2.VideoCapture(0)
try:
    for i in result:
        print(i.summary())
        if cv2.waitKey()==ord('q'):
            break
    cv2.destroyAllWindows()
    cam.release() # close camera
except:
    cv2.destroyAllWindows()
    cam.release()
