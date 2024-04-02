import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

#creating object
cap=cv2.VideoCapture(0)
#set frame height->480,width->640
cap.set(3,640)
cap.set(4,480)
segmentor=SelfiSegmentation(model=1)

listImg=os.listdir("BackgroundImages")
print(listImg)
imgList=[]
for imgPath in listImg:
    img=cv2.imread(f'BackgroundImages/{imgPath}')
    imgList.append(img)
print(len(imgList))

indexImg=0

while True:
    success, img=cap.read()
    imgOut=segmentor.removeBG(img,imgList[indexImg],cutThreshold=0.8)

    imgStacked=cvzone.stackImages([img,imgOut],2,1)
    print(indexImg)
    cv2.imshow("Image",imgStacked)
    
    key=cv2.waitKey(1)
    if key==ord('a'):
        if indexImg>0:
            indexImg-=1
    elif key==ord('d'):
        if indexImg<len(imgList)-1:
            indexImg+=1
    elif key==ord('q'):
        break