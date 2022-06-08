import cv2
import matplotlib.pyplot as plt
import numpy as np 


name = "none"

image = cv2.imread("./{}.jpg".format(name)) 
image =cv2.resize(image,(400,400))
image = cv2.blur(image,(5,5))



copy = image.copy()
histogram = cv2.cvtColor(copy,cv2.COLOR_BGR2HSV)
copy = np.array(histogram)

print(copy.shape)
hue,saturation,value = copy[:,:,0],copy[:,:,1],copy[:,:,2]

plt.title("{} Hue Histogram".format(name.capitalize()))
plt.hist(hue.ravel(),360,[0,360]); plt.show()

cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
cv2.imshow('custom window', copy)
cv2.resizeWindow('custom window', 600, 600)


cv2.waitKey(0)