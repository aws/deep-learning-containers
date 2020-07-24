import cv2
import numpy as np

IMG_PATH='/test_img.jpg'


def test_transformation(img):
    # Scaling
    res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    # Translation
    rows,cols = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    print('Successfully test OpenCV with image transformation')


if __name__ == '__main__':
    img = cv2.imread(IMG_PATH,0)
    print('CVtest: load the image')
    test_transformation(img)
