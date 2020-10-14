import numpy as np
import cv2
import os


def getDigits(imgFolderPath, num = 216):
    digits = np.zeros((num,232, 412), dtype = 'uint8') # digits.shape[1:] corresponds to greyscale card images
    # digits = [None for a in range(num)] list method
    targets = np.zeros(num, dtype = 'uint8')

    directory = os.fsencode(imgFolderPath)

    for inDex,file in enumerate(os.listdir(directory)[:num]):
         filename = os.fsdecode(file)
         # print(filename)
         if filename.endswith(".jpg") and 'W' not in filename:
             # print(os.path.join(directory, filename))
             image = cv2.imread(imgFolderPath + "\\" + filename)
             if image.shape[1] == 4128:
                 digits[inDex] = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), (int(image.shape[1]/10), int(image.shape[0]/10)), interpolation = cv2.INTER_AREA)
             elif image.shape[1] == 2322: # for alternate shapes, directing towards common digits numpy array
                 digits[inDex] = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), (int(image.shape[0]/10), int(image.shape[1]/10)), interpolation = cv2.INTER_AREA)
             targets[inDex] = (inDex // 4) + 1


    return (digits, targets)



if __name__ == '__main__':
    (data, targets) = getDigits("C:\\Users\\Tamara\\Desktop\\sklearnpractice\\img",5)
