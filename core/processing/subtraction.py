from skimage.metrics import structural_similarity
import cv2 as cv
import numpy as np

def differenceInspection(baseimg, tocompareimg, showResults=False):

    height, width, _ = baseimg.shape

    tocompareimg = cv.resize(tocompareimg, (width, height))

    # Convert images to grayscale
    before_gray = cv.cvtColor(baseimg, cv.COLOR_BGR2GRAY)
    after_gray = cv.cvtColor(tocompareimg, cv.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] baseimg we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(baseimg.shape, dtype='uint8')
    filled_after = tocompareimg.copy()

    for c in contours:
        area = cv.contourArea(c)
        if area > 40:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(baseimg, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv.rectangle(tocompareimg, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv.drawContours(mask, [c], 0, (0, 255, 0), -1)
            cv.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    if showResults:

        cv.imshow('diff', diff)
        cv.imshow('mask', mask)
        cv.imshow('filled after', filled_after)

    return diff, mask, filled_after