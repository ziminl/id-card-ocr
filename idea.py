


import cv2
import pytesseract



image = cv2.imread('image.jpg')



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
text = pytesseract.image_to_string(gray)

print(text)


