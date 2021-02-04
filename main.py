import image, sensor, framebuf, time, random, math, gc
from ulab import numpy as np

sensor.shutdown(True)
gc.enable()


##### MATH #####
def addNum(img, num):
    myImg = img.copy()
    for x in range(myImg.width()):
        for y in range(myImg.height()):
            newVal = myImg.get_pixel(x,y)+num
            if newVal<0: newVal=0
            if newVal>255: newVal=255
            myImg.set_pixel(x, y, newVal)
    return myImg

##### UTIL #####
def resetImage(img_path):
    myImage = image.Image(imgPath, copy_to_fb = True)
    myImage.to_grayscale(copy_to_fb=True)
    return myImage

def ndarrayToImage(arr, doesCopy):
    shape = arr.shape()

    if len(shape) == 1:
        width = shape[0]
        height = 1
    else:
        height, width = shape
    newImg = image.Image(width, height, sensor.GRAYSCALE, copy_to_fb=doesCopy)
    newImg.to_grayscale(copy_to_fb=doesCopy)

    if height==1:
        for i in range(width):
            newImg[i] = int(math.ceil(arr[i]))
    else:
        for x in range(width):
            for y in range(height):
                newImg.set_pixel(x, y, int(math.ceil(arr[y][x])))

    sensor.flush()
    time.sleep_ms(1000)
    return newImg

def imgToArray(img):
    return np.array(img, dtype=np.uint8)

def printSig(img):
    ar = imgToArray(img)
    for i in range(len(ar[0])):
        print(ar[0][i])

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
            #print(randX, randY)
            #print(img)
            square = img.copy(roi=(randX, randY, sqSize, sqSize), copy=False)
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
    return k

##### GRIDDING #####
def circleHash(c):
    string = str(c.x())+"-"+str(c.y())
    return string

def getCircles(img):
    circles = []
    width = img.width()
    height = img.height()
    totX = width // 100
    totY = height // 100

    for x in range(totX):
        for y in range(totY):
            x1 = x*100
            y1 = y*100
            if x1<(width-100):
                x2 = 200
            else:
                x2 = width-x1-1
            if y1<(height-100):
                y2 = 200
            else:
                y2 = height-y1-1
            _roi=(x1,y1,x2,y2)

            circles = circles + img.find_circles(roi=_roi, r_max=12, x_margin=12, y_margin=12, r_margin=12)

    newCircles = []
    newSet = set()

    for c in circles:
        h = circleHash(c)
        if h not in newSet:
            newSet.add(h)
            t_roi=(c.x()-c.r()+1, c.y()-c.r()+1, 2*(c.r()-1), 2*(c.r()-1))
            print(t_roi)
            print(img)
            t_square=img.copy().crop(copy=True, roi=t_roi)
            print(t_square)
            if t_square.get_statistics().mean()>170:
                newCircles.append(c)
            del t_square

    return newCircles

def getLines(img, circles, numRows, numCols):
    minX=img.width()-1
    maxX=0
    minY=img.height()-1
    maxY=0

    for c in circles:


def gridding(img):
    circles = getCircles(img)
    for c in circles:
        print(c)
    print(len(circles))

###########################
##### MAIN #####
#ogPath = "/sample.bmp"
imgPath = "/mysample.bmp"

myImage = image.Image(imgPath, copy_to_fb = True)
myImage.to_grayscale(copy_to_fb=True)

preprocess(imgPath, myImage)
myImage.save("/contrasted.bmp")
myImage.scale()
gc.collect()
print(gc.mem_free())
sensor.flush()
time.sleep_ms(2000)

gridding(myImage)



