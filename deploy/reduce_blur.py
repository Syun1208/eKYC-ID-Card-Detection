import cv2
import numpy as np
from PIL import Image


# kernel = np.ones((2, 2), np.uint8)
# alignedImage = cv2.dilate(alignedImage, kernel, iterations=1)
# alignedImageGray = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2GRAY)
# alignedImageBlur = cv2.GaussianBlur(alignedImageGray, (0, 0), sigmaX=33, sigmaY=33)
# alignedImage = cv2.addWeighted(alignedImageBlur, 1.5, alignedImageGray, -0.5, 0, alignedImageGray)
# alignedImageDivide = cv2.divide(alignedImageGray, alignedImageBlur, scale=255)
# thresh = cv2.threshold(alignedImageDivide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# alignedImage = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
# alignedImage = cv2.bitwise_not(alignedImage, alignedImage)
# alignedImage = cv2.Canny(alignedImage, 50, 100)


class reduceBlur:
    def __init__(self, image):
        self.image = image

    def sharpenImage(self):
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(self.image, -1, sharpen_kernel)
        return sharpen

    def convert2BinaryImage(self):
        kernel = np.ones((2, 2), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        self.image = cv2.Canny(morph, 50, 100)
        return self.image

    def checkBlurriness(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        text = "Not Blurry"
        if fm < 80:
            text = "Blurry"
        cv2.putText(self.image, "{}: {:.2f}".format(text, fm), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        return self.image

    def brightnessContrast(self, brightness=150, contrast=210):
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                maximum = 255
            else:
                shadow = 0
                maximum = 255 + brightness
            al_pha = (maximum - shadow) / 255
            ga_mma = shadow
            cal = cv2.addWeighted(self.image, al_pha, self.image, 0, ga_mma)
        else:
            cal = self.image
        if contrast != 0:
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
            Gamma = 127 * (1 - Alpha)
            cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)
        return cal

    def sharpeningImage(self):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image_sharp = cv2.filter2D(src=self.image, ddepth=-1, kernel=kernel)
        return image_sharp

    def __call__(self, *args, **kwargs):
        return self.brightnessContrast()

# if __name__ == '__main__':
#     brightness = int(input('brightness: '))
#     contrast = int(input('contrast: '))
#     image = cv2.imread(
#         '/home/long/Desktop/IDCardDetectionandRecognition/results/correct/310232042_509348944311892_3584048226966180568_n.jpg')
#     reduce = reduceBlur(image)
#     reduce.brightnessContrast(brightness, contrast)
#     cv2.imwrite(
#         '/home/long/Desktop/IDCardDetectionandRecognition/results/mask/310232042_509348944311892_3584048226966180568_n.jpg',
#         reduce())
