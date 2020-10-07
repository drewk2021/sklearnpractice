import numpy as np
import cv2
import os


def getDigits(imgFolderPath):
    digits = np.zeros((208,4128,2322), dtype = 'uint8') # digits.shape[1:] corresponds to greyscale card images
    targets = np.zeros(216, dtype = 'uint8')

    directory = os.fsencode(imgFolderPath)

    for inDex,file in enumerate(os.listdir(directory)):
         filename = os.fsdecode(file)
         if filename.endswith(".jpg") and 'W' not in filename:
             # print(os.path.join(directory, filename))
             image = cv2.imread(filename)
             digits[inDex] = image.cvtColor(image,cv2.COLOR_BGR2GRAY)
             targts[inDex] = (inDex // 4) + 1


    return (digits, targets)
