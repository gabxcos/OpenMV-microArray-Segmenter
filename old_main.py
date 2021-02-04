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

##### GRIDDING #####
def getProjections(mat):
    Hlines = np.mean(mat, axis=1)
    Vlines = np.mean(mat, axis=0)
    return (Hlines, Vlines)

def calculateKernelSize(proj):
    length = len(proj)

    accum = 0
    numNonZero = 0

    wasZero = True
    for i in range(length):
        val = proj[i]
        isNonZero = val > 0
        if isNonZero: accum = accum+1
        if not isNonZero and not wasZero: numNonZero = numNonZero+1
        wasZero = not isNonZero

    return math.ceil(accum/numNonZero)

def getReconstruction(marker, mask, kernelSize):
    size = max(2, min(kernelSize, 21))

    m1 = marker.copy()
    for i in range(15):
        m0 = m1.copy()
        m1.dilate(size).min(mask)
        m0.difference(m1)
        empty = True
        for x in range(m0.width()):
            if not empty: break
            for y in range(m0.height()):
                if m0.get_pixel(x,y)!=0:
                    empty=False
                    break

        if empty: break

    return m1

def calculateSignals(projections, kernelSizes):
    H, V = projections
    kernelH, kernelV = kernelSizes
    Hmean = H.get_statistics().mean()
    Vmean = V.get_statistics().mean()

    _H = addNum(H, -Hmean)
    _V = addNum(V, -Vmean)

    Hrec = getReconstruction(_H, H, kernelH)
    Vrec = getReconstruction(_V, V, kernelV)

    H_mark = H.copy().sub(Hrec)
    V_mark = V.copy().sub(Vrec)

    return (H_mark, V_mark)

def getBinarySignals(signals):
    Hsig, Vsig = signals
    Hthresh = Hsig.get_histogram().get_threshold().value()
    Vthresh = Vsig.get_histogram().get_threshold().value()

    Hbin = Hsig.copy().binary([(0,Hthresh)], invert=True)
    Vbin = Vsig.copy().binary([(0,Vthresh)], invert=True)
    return (Hbin, Vbin)

def getLines(binSig):
    spot = []
    lspot = []

    flag = True
    t = 0
    for i in range(binSig.width()):
        if flag:
            if binSig.get_pixel(i, 0)>254:
                spot.append(i)
                flag = not flag
                t = 0
        else:
            t = t+1
            if t>4 and binSig.get_pixel(i,0)<1:
                spot.append(i)
                flag = not flag

    lspot.append(spot[0])
    for i in range(1,len(spot)-1, 2):
        val = math.ceil((spot[i]+spot[i+1])/2)
        lspot.append(val)

    lspot.append(spot[len(spot)-1])

    return lspot

def adjustToDevice(img, lines):


def gridding(img):
    mat = imgToArray(img)
    Hproj, Vproj = getProjections(mat)

    Hkernel = calculateKernelSize(Hproj)
    Vkernel = calculateKernelSize(Vproj)

    HlinesImg = ndarrayToImage(np.array(Hproj, dtype=np.uint8), False)
    VlinesImg = ndarrayToImage(np.array(Vproj, dtype=np.uint8), False)

    Hsig, Vsig = calculateSignals((HlinesImg, VlinesImg),(Hkernel, Vkernel))

    Hbin, Vbin = getBinarySignals((Hsig, Vsig))

    Hlines = getLines(Hbin)
    Vlines = getLines(Vbin)

    print(Hlines, Vlines)

    return (Hbin, Vbin)

###########################
##### MAIN #####
#ogPath = "/sample.bmp"
imgPath = "/mysample.bmp"

myImage = image.Image(imgPath, copy_to_fb = True)
myImage.to_grayscale(copy_to_fb=True)

preprocess(imgPath, myImage)
myImage.save("/contrasted.bmp")
gc.collect()

Htest, Vtest = gridding(myImage)

#print(imgToArray(Htest), imgToArray(Vtest))

Htest.save("/Htest.bmp")
Vtest.save("/Vtest.bmp")
time.sleep_ms(2000)

