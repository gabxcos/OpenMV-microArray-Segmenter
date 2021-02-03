import image, sensor, framebuf, random, math
import ulab as np

sensor.shutdown(True)

##### UTIL #####
def resetImage(img_path):
    myImage = image.Image(imgPath, copy_to_fb = True)
    myImage.to_grayscale(copy_to_fb=True)
    return myImage

###### PREPROCESSING ######
def calculateK(img):
    width = img.width()
    height = img.height()
    sqSize = 15
    numIter = 300

    lowBound = img.get_statistics().mean()
    upBound = 0.9

    tempK = 0

    randX = random.randint(1, 500)
    randY = random.randint(1, 500)

    for _ in range(numIter):
        minMax = upBound
        for _ in range(12):
            random.seed(randX * randY)
            randX = random.randint(sqSize, width - sqSize - 1)
            randY = random.randint(sqSize, height - sqSize - 1)
            square = img.crop(roi=(randX, randY, sqSize, sqSize), copy=True)
            maxVal = square.get_statistics().max()
            if (maxVal < minMax) and (maxVal > lowBound) and (maxVal < upBound):
                minMax = maxVal
            del square
        tempK = tempK + minMax

    k = tempK / numIter
    return k


def calculateContrast(img_path, img):
    width = img.width()
    height = img.height()
    meanVal = img.get_statistics().mean()

    for x in range(width):
        for y in range(height):
            img.set_pixel(x, y, img.get_pixel(x, y) - meanVal)
    d2img = img.gamma_corr(0.25)
    d2 = d2img.get_statistics().mean()
    d = math.sqrt(d2)

    m4img = d2img.gamma_corr(0.5)
    del d2img
    m4 = m4img.get_statistics().mean()
    del m4img
    alfa4 = d / math.pow(d2, 2)
    C = d / pow(alfa4, (1.0/4.0))
    C = 10000.0/C

    return C

def contrastEnhance(img, C, k):
    width = img.width()
    height = img.height()
    for x in range(width):
        for y in range(height):
            currVal = img.get_pixel(x,y)
            if currVal >= k:
                newVal = int(currVal * C)
                if newVal > 255:
                    newVal = 255
                img.set_pixel(x,y, newVal)

def preprocess(img_path, img):
    k = calculateK(img)
    C = calculateContrast(img_path, img)
    del img
    img = resetImage(img_path)
    contrastEnhance(img, C, k)
    img = img.median(3)

###########################
##### MAIN #####
imgPath = "/mysample.bmp"
myImage = image.Image(imgPath, copy_to_fb = True)
myImage.to_grayscale(copy_to_fb=True)

width = myImage.width()
height = myImage.height()

preprocess(imgPath, myImage)
myImage.save("/contrasted.bmp")
myImage = image.Image("/contrasted.bmp", copy_to_fb=True)
