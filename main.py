from mss import mss
from PIL import Image
import cv2
import pytesseract
import time
from coordinates import defineURPoint,defineDLPoint,defineRefreshPoint

counter = 1
x1, y1 = defineURPoint()
x2, y2 = defineDLPoint()
x3, y3 = defineRefreshPoint()
while True:
    with mss() as sct:
        sct.shot()

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Andrey\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'
    im = Image.open('monitor-1.png')
    im_crop = im.crop((x1, y1, x2, y2))
    im_crop.save('MarketPic.jpg', quality=95)
    img = cv2.imread('MarketPic.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    config = r'--oem 3 --psm 6'

    l = list(pytesseract.image_to_string(img, config=config))[:5]

    if '\n' in l:
        l.remove('\n')

    prices = int(''.join(l))
    if (prices) <= 1001:
        print(prices)
    else:
        print(type(prices))
        print('Нет карт по выбранной цене')
    print(f"Проверка номер {counter}")
    print("-"*40)
    time.sleep(3)

'''
data = pytesseract.image_to_data(img, config=config)
for i, el in enumerate(data.splitlines()):
    if i == 0:
        continue

    el = el.split()
    try:
        x, y, w, h = int(el[6]), int(el[7]), int(el[8]), int(el[9])
        cv2.rectangle(img, (x, y), (w + x, h + y), (0, 0, 255), 1)
        cv2.putText(img, el[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    except IndexError:
        print("Операция была пропущена")

cv2.imshow('Result', img)
cv2.waitKey(0)
'''
