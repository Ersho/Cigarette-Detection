import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
# fileName = ['9','8','7','6','5','4','3','2','1','0']

image = cv2.imread('cigarette2.jpg')

def try_YCrCb(image):

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    plt.imshow(img[:,:,0],cmap = 'gray')
    plt.imshow(img[:,:,1],cmap = 'gray')
    plt.imshow(img[:,:,2],cmap = 'gray')
    
    ret,thresh1 = cv2.threshold(img[:,:,1],150,255,cv2.THRESH_BINARY)
    blur = cv2.medianBlur(thresh1,5)
    plt.imshow(blur, cmap = 'gray')
    
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(blur,kernel,iterations = 2)
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 2)
    
    plt.imshow(dilation, cmap = 'gray')
    
    index = 0
    while os.path.exists('output-dilation-img-{}.jpg'.format(index)):
        index +=1
    cv2.imwrite('output-dilation-img-{}.jpg'.format(index), dilation)
    
    edged = cv2.Canny(dilation, 30, 200)
    
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    
    cv2.imwrite('output-img-{}.jpg'.format(index), image)

def mask_approach(image):
    img = image
    
    white_mask = cv2.inRange(img, (230,230,230), (255,255,255))
    blur = cv2.medianBlur(white_mask,5)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(blur,kernel,iterations = 2)
    
    edged = cv2.Canny(dilation, 30, 200)
    
    _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    conts2 = []
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    for index,rect in enumerate(rects):
        x,y,w,h = (rect)
        area = w * h
        if area > 300 and area < 1800:
            conts2.append(contours[index])
    cv2.drawContours(image, conts2, -1, (0,255,0), 3)
    
    cv2.imwrite('output-both-img.jpg',image)
    
    plt.imshow(dilation, cmap = 'gray')

