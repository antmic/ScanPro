import os
import cv2
import numpy as np

# DEFINITIONS

def open():
    img = cv2.imread(path + item.name, flags=1)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayimg

def binarize(image):

    #def resize(image):
    #    img = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))
    #    return img

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
    
    #img_resized = resize(image)
    #show(img_resized, "resized")
    img_filtered = bil_filter(image)
    #show(img_filtered, "filtered")
    img_closed = closing(image)
    #show(img_closed, "closed")
    img_divided = divide(img_filtered, img_closed)
    #show(img_divided, "divided")
    img_o = otsu(img_divided)
    #show(img_o, "otsu")
    #img_ag = adapt_gauss(img_divided)
    #show(img_ag, "adaptive gauss")

    return img_o

def show(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ACTIONS

path = "/home/antoni/Dokumenty/PythonProjects/ScanPro/test/" #input("Enter path to folder: ")
with os.scandir(path) as folder:
    for item in folder:
        if item.name.lower().endswith('.jpg'):
            
            img = open()
            
            img_bin = binarize(img)
            
            img_name = os.path.splitext(item.name)[0]
            cv2.imwrite(path + img_name + "_filtered.jpg", img_bin)
            
        else:
            print(item.name," unrecognized")
            continue