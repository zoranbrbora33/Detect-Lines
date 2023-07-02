import cv2
import numpy as np

img = cv2.imread('autocesta.jpeg')
img_blurred = cv2.GaussianBlur(img, (7, 7), None)
img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
img_canny = cv2.Canny(img_gray, 50, 200)

mask = np.zeros_like(img_canny)
cv2.ellipse(img=mask,
            center=(mask.shape[1] // 2, mask.shape[0]),
            axes=(mask.shape[1] // 2, mask.shape[0] // 2),
            angle=0,
            startAngle=180,
            endAngle=360,
            color=255,
            thickness=-1
            )

filtered_canny = cv2.bitwise_and(img_canny, mask)

lines = cv2.HoughLinesP(filtered_canny, 2, np.pi / 180,
                        300, minLineLength=100, maxLineGap=10)

for line in lines:
    line = line.squeeze()
    cv2.line(img, (line[0], line[1]), (line[2], line[3]),
             (0, 255, 0), thickness=3)


cv2.imshow('lines', img)

cv2.waitKey()
cv2.destroyAllWindows()
