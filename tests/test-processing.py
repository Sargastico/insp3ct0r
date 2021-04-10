import cv2 as cv
import random as rng
import imutils
import core

# ============== OPTIONS =================#

debug = True  # Set "True" to enable display of debug info/images
dump = False  # Set "True" to enable dump of debug info/images

base_image = 'baseimage.jpg'

# ========================================#

rng.seed(12345)


def main():
    cap = cv.VideoCapture(1)
    ret, src = cap.read()

    if ret:

        cv.imwrite("capture.jpg", src)
        src = cv.imread("capture.jpg")

        originalCapture = cv.imread("capture.jpg", cv.IMREAD_COLOR)
        refFilename = "baseImage.jpg"
        imReference = cv.imread(refFilename, cv.IMREAD_COLOR)

        imReg, h, imMatches = core.processing.homography.HomographyCorrection(src, imReference)
        draw, thresh_im, _ = core.processing.utils.drawRoi(src, True, False)
        imagePca, angleDegree = core.processing.pca.pcaTransform(src, False)

        rotationImage = imutils.rotate(originalCapture, angle=angleDegree)
        cv.imwrite('fixedorientation.jpg', rotationImage)

        print("Estimated homography : \n", h)

        dst = core.processing.utils.imageOverlay(src, draw)

        if debug:
            cv.imshow("perspective", imReg)
            cv.imshow("rotated", rotationImage)
            cv.imshow('threshold', thresh_im)
            cv.imshow("features", imMatches)
            cv.imshow("bounding", dst)

        if dump:
            cv.imwrite("perspective.jpg", src)
            cv.imwrite("rotated.jpg", rotationImage)
            cv.imwrite("threshold.jpg", thresh_im)
            cv.imwrite("feature.jpg", imMatches)
            cv.imwrite("bounding.jpg", dst)

        fixed = cv.imread('fixedorientation.jpg')
        draw, thresh_im, cropped_image = core.processing.utils.drawRoi(fixed, True, True)
        dst = core.processing.utils.imageOverlay(fixed, draw)

        cv.imshow('ROI after orientation fix', dst)
        cv.imwrite('croppedimage.jpg', cropped_image)

        baseimage = cv.imread(base_image)
        draw, thresh_im, cropped_image = core.processing.utils.drawRoi(baseimage, True, True)
        dst = core.processing.utils.imageOverlay(baseimage, draw)

        cv.imshow('ROI base image', dst)

        cv.imwrite('croppedbaseimage.jpg', cropped_image)

        img1 = cv.imread('croppedbaseimage.jpg')
        img2 = cv.imread('croppedimage.jpg')

        diff, mask, filled_after = core.processing.subtraction.differenceInspection(img1, img2, True)

        _, binarized_diff = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        if dump:
            cv.imwrite('binarized_diff.jpg', binarized_diff)
            cv.imwrite('diff.jpg', diff)
            cv.imwrite('mask.jpg', mask)
            cv.imwrite('filled_after.jpeg', filled_after)

    cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()


main()
