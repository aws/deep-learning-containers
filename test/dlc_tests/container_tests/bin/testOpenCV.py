import cv2
import numpy as np
import sys



def test_transformation(img):
    print("Path of the image: ", img)
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions: ', img.shape)

    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Scaling
    res = cv2.resize(img,dim,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    # Translation
    rows,cols,_= img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    print('Successfully test OpenCV with image transformation')


if __name__ == '__main__':
    print('CVtest: load the image')
    test_transformation(sys.argv[1])
