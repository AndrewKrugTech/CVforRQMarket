from turtle import width
import cv2
import numpy as np
from PIL import Image, ImageOps
import math


def repainting(image_file):
    splitDigits()
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_array = np.array(gray)

    for i in range(np_array.shape[0]):
        for j in range(np_array.shape[1]):
            if np_array[i][j] < 35:
                np_array[i][j] = 255
            else:
                np_array[i][j] = 0
    
    pil_image = Image.fromarray(np_array)
    pil_image.save('repainted.png', format="png")


def splitDigits():
    list = []
    #unitedList = []
    k = 0
    image_file = "price_test.png"
    im = Image.open(image_file)
    width, height = im.size

    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_array = np.array(gray)

    y = np_array.shape[0]
    x = np_array.shape[1]
    x_1 = math.floor(y / 2)
    x_2 = x_1 - 3
    x_3 = x_1 + 3
    
    #(np_array[x_1][j] > 35 and np_array[x_2][j] > 35)
    # or (np_array[x_2][j] > 35 and np_array[x_3][j] > 35)
    for j in range(x):
        if (np_array[x_1][j] > 35 or np_array[x_2][j] > 35 or np_array[x_3][j] > 35) and k <= 0:
            image = (im.crop((j-1, 0, j + 6, y)))
            list.append(np.array(ImageOps.expand(image, border = 3, fill = 'black')))
            k = 6
        else:
            k -= 1
    
    del list[:2]
    #width += 1 * len(list)
    new_im = Image.new('RGB', (width, height))
    
    x_offset = 0    
    for imgs in list:
        imgs = Image.fromarray(imgs)
        new_im.paste(imgs, (x_offset,0))
        x_offset += imgs.size[0]
    
    new_im.save('test.png')


repainting('test.png')

