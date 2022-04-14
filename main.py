import pyautogui
from mss import mss
from PIL import Image
import cv2
import pytesseract
import time
from coordinates import definePoints
import keyboard as kb

def picsize():
    im = Image.open("monitor-1.png")
    (x, y) = im.size
    return x,y

counter = 1
i = 0
x1, y1, x2, y2, x3, y3 = definePoints()

while True:# kb.is_pressed("F9") != True:
    with mss() as sct:
        sct.shot()

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    im = Image.open('monitor-1.png')
    im_crop = im.crop((x1, y1, x2, y2))
    im_crop.save('MarketPic.jpg', quality=95)
    img = cv2.imread('MarketPic.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    config = r'--oem 3 --psm 6'
    l = list(pytesseract.image_to_string(img, config=config))  # [:5]
    #print(l)
    #print(i, l)
    correctedList = l[:l.index('\n')]
    print(correctedList)

    while '©' in correctedList:
        correctedList.remove('©')
    while '.' in correctedList:
        correctedList.remove('.')
    while ' ' in correctedList:
        correctedList.remove(' ')
    while '@' in correctedList:
        correctedList.remove('@')
    print(correctedList)

    try:
        price = int(''.join(correctedList))
    except ValueError:
        price = 1001
    else:
            if price <= 1000:
                #print(price)
                pyautogui.click(x=x1-1590, y=y1+190, clicks=6)
                for i in range(5):
                    kb.send("Enter")
                print(f"Цена карты составила {price}")
            else:
                #print(type(price), price)
                print('Нет карт по выбранной цене')
                pyautogui.click(x=x3-1600, y=y3+180, clicks=1)
    print(f"Проверка номер {counter}. Минимальная цена {price}")
    counter += 1
    print("-" * 40)
    time.sleep(3)

"""
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
"""
