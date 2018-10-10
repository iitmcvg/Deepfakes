import cv2

file ="51827818.jpg"
img = cv2.imread(file,0)
print(img.shape)
resize_img = cv2.resize(img  , (64 , 64))
print(resize_img.shape)
