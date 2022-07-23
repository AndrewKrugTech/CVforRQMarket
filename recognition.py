import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from tensorflow import keras
from typing import *
import time
import math
from PIL import Image, ImageOps


def cnn_print_digit(d):
    print(d.shape)
    for x in range(28):
        s = ""
        for y in range(28):
            s += "{0:.1f} ".format(d[28*y + x])
        print(s)


def cnn_print_digit_2d(d):
    print(d.shape)
    for y in range(d.shape[0]):
        s = ""
        for x in range(d.shape[1]):
            s += "{0:.1f} ".format(d[x][y])
        print(s)


emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]

def emnist_predict(model, image_file):
    img = keras.preprocessing.image.load_img(image_file, target_size=(28, 28), color_mode='grayscale')
    emnist_predict_img(model, img)


def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(emnist_labels[result[0]])

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


def splitDigits(): #split digits from picture, need to fix
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
    
    del list[:2] #deleting "Gold" sign
    
    #width += 1 * len(list)
    new_im = Image.new('RGB', (width, height))
    
    x_offset = 0    
    for imgs in list:
        imgs = Image.fromarray(imgs)
        new_im.paste(imgs, (x_offset,0))
        x_offset += imgs.size[0]
    
    new_im.save('test.png')

def letters_extract(image_file: str, out_size=28):
    
    image_file = "repainted.png"
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(  
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) 
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)



    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            
            #resize letter canvas to square
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                #enlarge image top-bottom
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                #enlarge image left-right
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            #resize letter to 28x28 and add letter and its X-coordinate
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    #sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


def img_to_str(model: Any, image_file: str):
    letters = letters_extract(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out


if __name__ == "__main__":

    start_time = time.time()
    model = keras.models.load_model('emnist_letters.h5')
    s_out = img_to_str(model, "price_test.png")
    print(s_out)
    print("Execution time %s seconds" % (time.time() - start_time)) 