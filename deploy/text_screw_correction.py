import argparse
import numpy as np
import cv2


class textScrewCorrection:
    def __init__(self, image):
        self.image = image
        self.height = image.shape[0]
        self.width = image.shape[1]

    def __call__(self, *args, **kwargs):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        center = (self.width // 2, self.height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (self.width, self.height), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        print("[INFO] angle: {:.3f}".format(angle))
        cv2.imshow("Input", self.image)
        cv2.imshow("Rotated", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
