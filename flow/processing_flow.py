import cv2 as cv
import random as rng
import imutils

from core import processing as proc

from core.processing.homography import HomographyCorrection
from core.processing.pca import pcaTransform
from core.processing.subtraction import differenceInspection
from core.processing.utils import drawRoi

rng.seed(12345)

def processFrame(srcFrame, refImage, optDebug=False, optDump=False):
    '''
    :param srcFrame: Image Frame to process
    :param refImage: Reference Image for Homography and Subtraction
    :param optDebug: Set 'True' to enable any Debug output
    :param optDump: Set 'True' to enable any file dump for debugging
    '''

    imReference = cv.imread(refImage, cv.IMREAD_COLOR)

    imReg, h, imMatches = HomographyCorrection(srcFrame, imReference)
    draw, thresh_im, _ = drawRoi(srcFrame, True)
    imagePca, angleDegree = pcaTransform(srcFrame)
    rotationImage = imutils.rotate(srcFrame, angle=angleDegree)
    cv.imwrite('assets/fixedorientation.jpg', rotationImage)

    if optDebug:
        print("Estimated homography : \n", h)

    dst = proc.utils.imageOverlay(srcFrame, draw)

    if optDebug:
        cv.imshow("perspective", imReg)
        cv.imshow("rotated", rotationImage)
        cv.imshow('threshold', thresh_im)
        cv.imshow("features", imMatches)
        cv.imshow("bounding", dst)

    if optDump:
        cv.imwrite("assets/perspective.jpg", srcFrame)
        cv.imwrite("assets/rotated.jpg", rotationImage)
        cv.imwrite("assets/threshold.jpg", thresh_im)
        cv.imwrite("assets/feature.jpg", imMatches)
        cv.imwrite("assets/bounding.jpg", dst)

    fixed = cv.imread('assets/fixedorientation.jpg')
    draw, thresh_im, cropped_image = proc.utils.drawRoi(fixed, True, True)
    dst = proc.utils.imageOverlay(fixed, draw)

    cv.imwrite('assets/croppedimage.jpg', cropped_image)

    baseimage = cv.imread(refImage)
    draw, thresh_im, cropped_image = proc.utils.drawRoi(imReference, True, True)

    cv.imwrite('assets/croppedbaseimage.jpg', cropped_image)

    img1 = cv.imread('assets/croppedbaseimage.jpg')
    img2 = cv.imread('assets/croppedimage.jpg')

    diff, mask, filled_after = proc.subtraction.differenceInspection(img1, img2)

    _, binarized_diff = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imwrite('assets/result.jpg', filled_after)

    if optDump:
        cv.imwrite('assets/binarized_diff.jpg', binarized_diff)
        cv.imwrite('assets/diff.jpg', diff)
        cv.imwrite('assets/mask.jpg', mask)
        cv.imwrite('assets/filled_after.jpeg', filled_after)

    return