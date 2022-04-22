import pyautogui
import keyboard as kb

"""
def defineURPoint():
    x1 = int()
    y1 = int()
    while y1 == 0:
        if kb.is_pressed("z"):
            x1, y1 = pyautogui.position()
    return x1, y1

def defineDLPoint():
    x2 = int()
    y2 = int()
    while y2 == 0:
        if kb.is_pressed("x"):
            x2, y2 = pyautogui.position()
    return x2, y2

def defineRefreshPoint():
    x3 = int()
    y3 = int()
    while y3 == 0:
        if kb.is_pressed("c"):
            x3, y3 = pyautogui.position()
    return x3, y3
"""


def definePoints():
    x1 = int()
    y1 = int()
    x2 = int()
    y2 = int()
    x3 = int()
    y3 = int()
    while y3 == 0:
        if kb.is_pressed("z"):
            x1, y1 = pyautogui.position()
            # z - правая верхняя точка прямоугольника цены лота
        elif kb.is_pressed("x"):
            x2, y2 = pyautogui.position()
            # y - левая нижняя точка прямоугольника цены лота
        elif kb.is_pressed("v"):
            x3, y3 = pyautogui.position()
            # v - точка обновления рынка
    return x1, y1, x2, y2, x3, y3
