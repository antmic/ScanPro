import os
import cv2 as cv
import numpy as np
#import timing

path = "/home/antoni/Dokumenty/PythonProjects/ScanPro/test/" #input("Enter path to folder: ")

# DEFINITIONS

def open():
    img = cv.imread(path + item.name, flags=1)
    grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return grayimg

def resize(image, divisor=4):
    img = cv.resize(image, (int(image.shape[1]/divisor), int(image.shape[0]/divisor)))
    return img

def binarize(image):

    def bil_filter(image):  
        img = cv.bilateralFilter(image, 5, 12, 12)
        return img

    def divide(image1, image2):
        img = image1/image2
        img = cv.normalize(img,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
        return img

    def closing(image):
        if image.shape[0] < 4000:
            kernel = np.ones((5,5),np.uint8)
        else:
            kernel = np.ones((7,7),np.uint8)
        closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=2)
        return closing

    #def adapt_gauss(image):
    #    img = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,9,6)
    #    return img

    def otsu(image):
        ret, img = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        return img
    
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

def show(image, name, resized=True, divisor=4):
    if resized == True:
        img = resize(image, divisor)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def save(image, path_to_image=path, name_addition="_filtered.jpg"):
    img_name = os.path.splitext(item.name)[0]
    cv.imwrite(path_to_image + img_name + name_addition, image)

# ACTIONS

with os.scandir(path) as folder:
    for item in folder:
        if item.name.lower().endswith('.jpg'):
            
            img = open()
            
            img_bin = binarize(img)
            
            show(img_bin, "test")
            
        else:
            print(item.name," file unrecognized")
            continue