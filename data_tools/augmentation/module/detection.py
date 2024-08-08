import cv2
import numpy as np

def detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 5, minLineLength=3, maxLineGap=5)

    if lines is not None:
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle = np.degrees(angle) % 180

            # if (25 <= angle <= 65 and x1 < 960) or (110 <= angle <= 155 and x1 > 960):
            if (10 <= angle <= 75 and x1 > 960) or (100 <= angle <= 175 and x1 < 960):
                filtered_lines.append(line[0])
    return filtered_lines

detection(cv2.imread('SDG.png'))