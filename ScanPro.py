import os
import cv2
import numpy as np

os.chdir("ScanProcessor/")

def open():
    img = cv2.imread('TestImages/' + item.name, flags=1)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayimg

def resize(image):
    img = image#cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))
    return img

def bil_filter(image):  
    img = cv2.bilateralFilter(image, 5, 12, 12)
    return img

def divide(image1, image2):
    img = image1/image2
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    return img

def closing(image):
    kernel = np.ones((7,7),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing

def adapt_gauss(image):
    img = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,6)
    return img

def otsu(image):
    ret, img = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img

def show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

with os.scandir('TestImages/') as folder:
    for item in folder:
        if item.name.lower().endswith('.jpg'):
            img_name = os.path.splitext(item.name)[0]
            
            img = open()
            
            img_resized = resize(img)
            
            closed = closing(img_resized)
            #show(closed, "closed")

            img_filtered = bil_filter(img_resized)
            #show(img_filtered, "filtered")

            img_divided = divide(img_filtered, closed)
            #show(img_divided, "divided")

            img_o = otsu(img_divided)
            #show(img_o, "otsu")

            #img_ag = adapt_gauss(img_divided)
            #show(img_ag, "adaptive gauss")

            cv2.imwrite("TestImages/" + img_name + "_filtered.jpg", img_o)
        else:
            print(item.name," unrecognized")
            continue