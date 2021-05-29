import cv2 as cv
import numpy as np
import random as rng


def imageOverlay(img1, img2):

    # create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # create a mask and create its inverse mask
    img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # black-out the area in ROI
    img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv.bitwise_and(img2, img2, mask=mask)

    # overlay
    dst = cv.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    return img1

def drawRoi(src, crop ,drawBoundingBoxes=False):

    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    threshold, thresh_im = cv.threshold(src_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    canny_output = cv.Canny(src_gray, threshold, threshold/2)

    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv.drawContours(drawing, contours_poly, hierarchy.max(), color)

    c = max(contours_poly, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(c)

    if drawBoundingBoxes:
        cv.rectangle(drawing, (x, y), (x + w, y + h), color, 2)

    cropped = None
    if crop:
        cropped = src[y:y+h, x:x+w]
        # cv.imshow('cropped', cropped)


    return drawing, thresh_im, cropped