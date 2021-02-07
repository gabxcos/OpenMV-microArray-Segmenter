import image, sensor, framebuf, time, random, math, gc, os
from ulab import numpy as np

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
def invert(ar):
    l = len(ar)
    _ar = []
    for i in range(l):
        _ar.append(ar[l-1-i])
    return _ar

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

def memoryFlush():
    gc.collect()
    print(gc.mem_free())
    sensor.flush()
    time.sleep_ms(2000)

##################################################################################################################################################################################################################
###### STARTUP ######
def hasInputDir():
    for d in os.ilistdir():
        if d[0]=='input' and d[1]==16384: return True
    return False

def getFiles():
    if hasInputDir():
        return [d[0] for d in os.ilistdir('/input') if d[1]==32768 and not d[0][0]=='.' and (d[0][-4:]=='.bmp' or d[0][-4:]=='.BMP')]    # file is file, not dir; file is not cache; file is bitmap
    else: return []

def manageOutput(inputFiles):
    # suppose file is "xyz.bmp", remove last 4 characters
    if 'output' not in os.listdir():
        os.mkdir('output')
    os.chdir('/output')
    currDirs = os.listdir()
    for f in inputFiles:
        name = f[0:-4]
        if name not in currDirs:
            os.mkdir(name)
    os.chdir('/')

def setup():
    inputFiles = getFiles()
    if len(inputFiles)>0:
        manageOutput(inputFiles)
    return inputFiles

###### PREPROCESSING ######
def calculateK(img):
    width = img.width()
    height = img.height()
    sqSize = 10
    numIter = 300

    lowBound = img.get_statistics().mean()
    upBound = 0.8*255

    tempK = 0

    randX = random.randint(1, 500)
    randY = random.randint(1, 500)

    for _ in range(numIter):
        minMax = upBound
        for _ in range(12):
            random.seed(randX * randY)
            randX = random.randint(sqSize, width - 2*sqSize - 1)
            randY = random.randint(sqSize, height - 2*sqSize - 1)
            #print(randX, randY)
            #print(img)
            square = img.copy(roi=(randX, randY, sqSize, sqSize), copy=False)
            maxVal = square.get_statistics().max()
            if (maxVal < minMax) and (maxVal > lowBound) and (maxVal < upBound):
                minMax = maxVal
            del square
        tempK = tempK + minMax

    k = tempK / numIter
    #print(k)
    return int(k)


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
    if d2==0: return 10
    alfa4 = d / math.pow(d2, 2)
    if alfa4==0: return 10
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

            if img.get_statistics(roi=t_roi).mean()>170:
                newCircles.append(c)

    return newCircles

def getLines(img, circles, numRows, numCols):
    displ = 5

    width = img.width()
    height = img.height()

    minX=width-1
    maxX=0
    minY=height-1
    maxY=0

    for c in circles:
        x = max(c.x()-c.r(),0)
        x2 = min(x+2*c.r(), width-1)
        y = max(c.y()-c.r(),0)
        y2 = min(y+2*c.r(), height-1)
        if x<minX: minX=x
        if x2>maxX: maxX=x2
        if y<minY: minY=y
        if y2>maxY: maxY=y2

    minX=max(0, minX-displ)
    maxX=min(width-1, maxX+displ)
    minY=max(0, minY-displ)
    maxY=min(height-1, maxY+displ)

    Hlines = [int(minY+(i*(maxY-minY)/numRows)) for i in range(numRows+1)]
    Vlines = [int(minX+(i*(maxX-minX)/numCols)) for i in range(numCols+1)]

    if Hlines[0]>Hlines[1]: Hlines=invert(Hlines)
    if Vlines[0]>Vlines[1]: Vlines=invert(Vlines)

    if (Vlines[numCols-1]-Vlines[0])<2*numCols or (Hlines[numRows-1]-Hlines[0])<2*numRows: return (None, None)

    return (Hlines, Vlines)

def gridding(img, numRows, numCols):
    circles = getCircles(img)
    Hlines, Vlines = getLines(img, circles, numRows, numCols)

    return Hlines, Vlines

def gridding_v2(img, numRows, numCols):
    width=img.width()-1
    height=img.height()-1
    Hlines = [int(i*height/numRows) for i in range(numRows+1)]
    Vlines = [int(i*width/numCols) for i in range(numCols+1)]
    return Hlines, Vlines


##### SEGMENTATION #####
def findSpotK(img, _roi):
    x,y,w,h = _roi
    #print(_roi)
    sqSize = min(min(3,w),h)
    numIter = 50

    lowBound = 0.3*255
    upBound = 0.9*255

    tempK = 0

    randX = random.randint(1, 500)
    randY = random.randint(1, 500)

    for _ in range(numIter):
        minMax = upBound
        for _ in range(12):
            random.seed(randX * randY)
            randX = random.randint(0, max(w - sqSize - 1,0))
            randY = random.randint(0, max(h - sqSize - 1,0))
            #print(randX, randY)
            #print(img)
            square = img.copy(roi=_roi, copy=False)
            maxVal = square.get_statistics(roi=(randX, randY, sqSize, sqSize)).max()
            if (maxVal < minMax) and (maxVal > lowBound) and (maxVal < upBound):
                minMax = maxVal
            del square
        tempK = tempK + minMax

    k = tempK / numIter
    #print(k)
    return int(k)

def segmentation(img, Hlines, Vlines):

    spots = []
    numRows = len(Hlines)-1
    numCols = len(Vlines)-1

    for y in range(numRows):
        spots.append([])
        for x in range(numCols):

            _roi=(Vlines[x], Hlines[y], Vlines[x+1]-Vlines[x], Hlines[y+1]-Hlines[y])

            if _roi[2]<=0 or _roi[3]<=0:
                print(Hlines)
                print(Vlines)

            k = findSpotK(img, _roi)
            blobs=img.find_blobs([(k,255)], merge=True, roi=_roi)

            if len(blobs)==0:
                spots[y].append(None)
            else:
                spots[y].append((blobs[0],k))

            del blobs

    return spots

##### QUALITY #####
def getAvgBlobSize(blobs):
    area = 0
    count = 0
    for l in blobs:
        for b in l:
            if b!=None:
                b=b[0]
                area = area + b.pixels()
                count = count+1

    if count>0: area = int(area/count)
    return area

def getSigBg(img, _blob, _roi):
    if _blob==None:
        signal = None
        bg=img.get_statistics([(0,255)],roi=_roi).median()
    else:
        b,k=_blob
        signal=img.get_statistics([(k,255)],roi=b.rect()).median()
        bg=img.get_statistics([(0,k)],roi=_roi).median()
    return (signal, bg)

def blobToQualityStepOne(img, _og_blob, device, row, col, Hlines, Vlines, avgSize):
    res = {}

    # general variables
    filled = not _og_blob==None

    _roi = (Vlines[col], Hlines[row], Vlines[col+1]-Vlines[col], Hlines[row+1]-Hlines[row])
    x,y,w,h = _roi

    recArea = w*h

    if filled:
        _blob,k=_og_blob
        blobArea=_blob.pixels()
    else:
        k=255
        blobArea=0
    bgArea=recArea-blobArea

    if filled:
        _blob_roi = _blob.rect()

        enclosing = _blob.enclosing_circle()
        enclosing_r = enclosing[2]
        enclosing_area = 3.14 * pow(enclosing_r, 2)

        enclosed = _blob.enclosed_ellipse()
        enclosed_r = min(enclosed[2], enclosed[3])
        enclosed_area = 3.14 * pow(enclosed_r, 2)


    # ROW+COL
    res["row"]=row+1
    res["col"]=col+1

    # ON
    signal, background = getSigBg(img,_og_blob,_roi)
    res["signal"]=signal
    res["background"]=background

    if filled:
        on = (signal - background) > 3 #(0.1*255)
    else:
        on = False
    res["on"]=on

    # DEVICE
    control = device[row][col]
    if control=="Hc":
        res["device"] = "HYB"
    if control=="Ec":
        res["device"] = "EMP"
    if control=="Nc":
        res["device"] = "NEG"
    if control=="Pc":
        res["device"] = "PCR"
    if control=="Cc":
        res["device"] = "CAP"

    # SIGNAL QUALITY
    if on:
        noiseBlobs=img.find_blobs([(signal,255)], merge=True, roi=_blob_roi)
        if len(noiseBlobs)>0:
            nonNoiseArea = noiseBlobs[0].pixels()
        else: nonNoiseArea=0
        noiseArea = blobArea-nonNoiseArea
        signalQuality=1-noiseArea/blobArea
        res["signalQuality"]=signalQuality
    else:
        res["signalQuality"]=None

    # BACKGROUND QUALITY
    noiseBlobs=img.find_blobs([(background,255)], merge=True, roi=_roi)
    if len(noiseBlobs)>0:
        nonNoiseArea = noiseBlobs[0].pixels()
    else: nonNoiseArea=0
    noiseArea = recArea-nonNoiseArea
    backgroundQuality=1-noiseArea/bgArea
    res["backgroundQuality"]=backgroundQuality

    # SCALE INVARIANCE
    if filled:
        scaleInvariance = enclosed_r/enclosing_r
    else:
        scaleInvariance = None
    res["scaleInvariance"]=scaleInvariance

    # SIZE REGULARITY
    if filled:
        sizeRegularity = enclosed_area/enclosing_area
    else:
        sizeRegularity = None
    res["sizeRegularity"]=sizeRegularity

    # SIZE QUALITY
    if filled:
        exponent = -abs(blobArea-avgSize)/blobArea
        sizeQuality = pow(math.e, exponent)
    else:
        sizeQuality = None
    res["sizeQuality"]=sizeQuality

    # SIGNAL-TO-NOISE RATIO
    if signal==None:
        snr = 1.0
    else:
        snr = signal/(signal+background)
    res["signalToNoiseRatio"]=snr

    # LOCAL BACKGROUND VARIABILITY
    bg_stats = img.get_statistics([(0,k)],roi=_roi)
    lbv = bg_stats.mean()/(bg_stats.stdev()+0.00001)
    res["localBackgroundVariability"]=lbv
    # to be updated in step two

    # LOCAL BACKGROUND HIGHNESS
    # done in step two

    # SATURATION QUALITY
    # !--- REMOVED

    # !--- NEW QUALITIES ---! #
    if filled:
        res["roundness"] = _blob.roundness()
    else:
        res["roundness"] = None

    # COMPOSITE QUALITY
    res["compositeQuality"] = res["signalToNoiseRatio"]
    if res["sizeQuality"]!=None:
        res["compositeQuality"] = res["compositeQuality"] * res["sizeQuality"]
    # to be updated in step two

    return res

def blobToQualityStepTwo(results):
    maxLBV = 0
    totBg = 0
    countBg = 0

    for res in results:
        LBV = res["localBackgroundVariability"]
        if LBV > maxLBV: maxLBV = LBV

        totBg = totBg + res["background"]
        countBg = countBg + 1

    if countBg>0:
        avgBg = totBg / countBg
    else:
        avgBg = 0

    mul = 1.0/maxLBV

    # update results
    for res in results:
        LBV = res["localBackgroundVariability"]
        res["localBackgroundVariability"] = LBV * mul

        background = res["background"]
        if avgBg<1:
            res["localBackgroundHighness"] = 1.0
        else:
            res["localBackgroundHighness"] = background / (background + avgBg)

        res["compositeQuality"] = res["compositeQuality"] * res["localBackgroundVariability"] * res["localBackgroundHighness"]

def getGeneralResults(results):
    HYB = [0,0, True]
    NEG = [0,0, True]
    PCR = [0,0, True]

    nullProperties = ["signal", "signalQuality", "scaleInvariance", "sizeRegularity", "sizeQuality", "roundness"]
    nonNullProperties = ["background", "backgroundQuality", "signalToNoiseRatio", "localBackgroundVariability", "localBackgroundHighness", "compositeQuality"]
    properties = nullProperties + nonNullProperties

    avg_res = {"signal":0, "background":0, "signalQuality":0, "backgroundQuality":0, "scaleInvariance":0, "sizeRegularity":0, "sizeQuality":0, "signalToNoiseRatio":0, "localBackgroundVariability":0, "localBackgroundHighness":0, "roundness":0, "compositeQuality": 0}
    std_dev_res = {"signal":0, "background":0, "signalQuality":0, "backgroundQuality":0, "scaleInvariance":0, "sizeRegularity":0, "sizeQuality":0, "signalToNoiseRatio":0, "localBackgroundVariability":0, "localBackgroundHighness":0, "roundness":0, "compositeQuality": 0}

    nonNull = 0
    tot = 0

    # CONTROLS + AVG

    for res in results:
        tot = tot + 1

        if res["device"]=="HYB":
            HYB[1]=HYB[1]+1
            if res["on"]==True:
                HYB[0]=HYB[0]+1
            else:
                HYB[2]=False

        if res["device"]=="PCR":
            PCR[1]=PCR[1]+1
            if res["on"]==True:
                PCR[0]=PCR[0]+1
            else:
                PCR[2]=False

        if res["device"]=="NEG":
            NEG[1]=NEG[1]+1
            if res["on"]==False:
                NEG[0]=NEG[0]+1
            else:
                NEG[2]=False

        hasNull = False
        for p in properties:
            if res[p]!=None:
                avg_res[p]=avg_res[p]+res[p]
            else:
                hasNull = True

        if not hasNull: nonNull = nonNull + 1

    controls = {
        "HYB":HYB,
        "NEG":NEG,
        "PCR":PCR
    }

    for p in nonNullProperties:
        avg_res[p] = avg_res[p]/tot

    for p in nullProperties:
        avg_res[p] = avg_res[p]/nonNull

    # STD DEV

    for res in results:


        for p in properties:
            if res[p]!=None:
                std_dev_res[p]=std_dev_res[p]+pow(avg_res[p]-res[p],2)

    for p in nonNullProperties:
        std_dev_res[p] = std_dev_res[p]/tot

    for p in nullProperties:
        std_dev_res[p] = std_dev_res[p]/nonNull

    return (controls, avg_res, std_dev_res)


def getResults(img, blobs, device, Hlines, Vlines):
    avgSize = getAvgBlobSize(blobs)
    results = []
    for y in range(len(blobs)):
        for x in range(len(blobs[0])):
            results.append(blobToQualityStepOne(img, blobs[y][x], device, y, x, Hlines, Vlines, avgSize))

    blobToQualityStepTwo(results)

    #for res in results: print(res)

    controls, avg_res, std_dev_res = getGeneralResults(results)

    return (results, controls, avg_res, std_dev_res)


##### RESULTS #####
def floatToString(f):
    if f==None: return 'NULL'
    else: return '%.2f' % f

def boolToString(b):
    if b: return '1'
    else: return '0'

def printProperty(results, numRows, numCols, prop, callback):
    text = ""
    for y in range(numRows):
        for x in range(numCols):
            res = results[y*numCols+x]
            text = text + callback(res[prop]) + "\t"
        text = text + "\n"

    return text

def printOnFile(myPath, text):
    f = open(myPath, "w")
    f.write(text)
    f.close()

def printSignal(sigPath, results, numRows, numCols):
    printOnFile(sigPath, printProperty(results, numRows, numCols, "signal", floatToString))

def printBackground(bgPath, results, numRows, numCols):
    printOnFile(bgPath, printProperty(results, numRows, numCols, "background", floatToString))

def printControl(name, control):
    text = name+" control: "
    if not control[2]: text = text + "NOT "
    text = text + "VALID"
    text = text + " : " + str(control[0]) + " spots out of " + str(control[1]) + "\n"
    return text

def printAvgStats(s):
    text = floatToString(s["signalQuality"])+"\t"
    text = text + floatToString(s["backgroundQuality"])+"\t"
    text = text + floatToString(s["scaleInvariance"])+"\t"
    text = text + floatToString(s["sizeRegularity"])+"\t"
    text = text + floatToString(s["sizeQuality"])+"\t"
    text = text + floatToString(s["signalToNoiseRatio"])+"\t"
    text = text + floatToString(s["localBackgroundVariability"])+"\t"
    text = text + floatToString(s["localBackgroundHighness"])+"\t"
    text = text + floatToString(s["compositeQuality"])+"\t"
    text = text + floatToString(s["roundness"])+"\n"

    return text

def printResults(resPath, results, controls, avg_res, std_dev_res, numRows, numCols, correctGridding):
    text = printProperty(results, numRows, numCols, "on", boolToString) + "\n"
    text = text + printControl("Hybridization", controls["HYB"])
    text = text + printControl("Negative", controls["NEG"])
    text = text + printControl("PCR", controls["PCR"]) + "\n"
    if not correctGridding: text = text + "Could not auto-grid correctly.\n\n"
    text = text + "Spot statistics:\n"
    text = text + "STAT\tSIG_Q\tBG_Q\tSCALE_I\tSIZE_R\tSIZE_Q\tSIGN_R\tLOCBG_V\tLOCBG_H\tCOMP_Q\tROUND\n"
    text = text + "AVG\t"+printAvgStats(avg_res)
    text = text + "STDEV\t"+printAvgStats(std_dev_res)
    printOnFile(resPath, text)

def printStat(res):
    text = str(res["row"])+"\t"+str(res["col"])+"\t"+boolToString(res["on"])+"\t"+res["device"]
    for p in ["signalQuality", "backgroundQuality", "scaleInvariance", "sizeRegularity", "sizeQuality", "signalToNoiseRatio", "localBackgroundVariability", "localBackgroundHighness", "compositeQuality", "roundness"]:
        text = text + "\t" + floatToString(res[p])
    text = text + "\n"
    return text

def printStatistics(statPath, results):
    f = open(statPath, "w")
    text = "ROW\tCOL\tON\tDEVICE\tSIG_Q\tBG_Q\tSCALE_I\tSIZE_R\tSIZE_Q\tSIGN_R\tLOCBG_V\tLOCBG_H\tCOMP_Q\tROUND\n"
    f.write(text)
    for res in results:
        f.write(printStat(res))
    f.close()


def totalPrint(name, results, controls, avg_res, std_dev_res, numRows, numCols, correctGridding):
    base = "/output/"+name+"/"
    printSignal(base+"signal.txt", results, numRows, numCols)
    gc.collect()
    printBackground(base+"background.txt", results, numRows, numCols)
    gc.collect()
    printResults(base+"results.txt", results, controls, avg_res, std_dev_res, numRows, numCols, correctGridding)
    gc.collect()
    printStatistics(base+"statistics.txt", results)
    gc.collect()


    # # # # # # # # #
    # # # TO DO # # #
    # # # # # # # # #

    # CAMBIARE METODO DI CALCOLO DEL K IN SEGMENTAZIONE?

##### TEST #####
def testAngles(results, numRows, numCols):
    totAngles = 0
    if results[0]["on"]: totAngles = totAngles + 1
    if results[numCols-1]["on"]: totAngles = totAngles + 1
    if results[numCols*(numRows-1)]["on"]: totAngles = totAngles + 1
    if results[numCols*numRows-1]["on"]: totAngles = totAngles + 1
    if totAngles<3: return False
    else: return True

##################################################################################################################################################################################################################
##### MAIN #####

# PREAMBLE
sensor.shutdown(True)
gc.enable()

# STARTUP

inputFiles = setup()

#define Hc Controls::HYBRIDIZATION
#define Ec Controls::EMPTY
#define Nc Controls::NEGATIVE
#define Pc Controls::PCR
#define Cc Controls::CAPTURE

device = [
[ "Hc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Ec", "Nc", "Ec", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Hc" ],
[ "Cc", "Cc", "Cc", "Pc", "Nc", "Cc", "Cc", "Cc", "Cc", "Ec", "Ec", "Cc", "Cc", "Cc", "Pc", "Nc", "Cc", "Cc", "Cc", "Cc", "Ec" ],
[ "Ec", "Cc", "Cc", "Cc", "Cc", "Hc", "Cc", "Cc", "Cc", "Ec", "Ec", "Ec", "Cc", "Cc", "Cc", "Cc", "Hc", "Cc", "Cc", "Cc", "Ec" ],
[ "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Ec", "Ec", "Nc", "Ec", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Ec", "Ec", "Nc" ],
[ "Nc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Ec", "Ec", "Hc", "Ec", "Nc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Ec", "Ec", "Hc" ],
[ "Hc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Ec", "Nc", "Ec", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Cc", "Hc" ]
]

numRows = len(device)
numCols = len(device[0])

for f in inputFiles:
    imgPath = "/input/"+f
    name = f[0:-4]

    print("\n")
    print(name)

    myImage = image.Image(imgPath, copy_to_fb = True)
    myImage.to_grayscale(copy_to_fb=True)

    # PREPROCESSING
    k = preprocess(imgPath, myImage)
    myImage.save("/output/"+name+"/contrasted.bmp")
    memoryFlush()

    # GRIDDING
    Hlines, Vlines = gridding(myImage, numRows, numCols)
    del myImage

    if Hlines==None or Vlines==None:
        _testAngles = False
    else:
        myImage = image.Image(imgPath, copy_to_fb = True)
        myImage.to_grayscale(copy_to_fb=True)
        memoryFlush()

        # SEGMENTATION
        spots = segmentation(myImage, Hlines, Vlines)

        # QUALITY
        results, controls, avg_res, std_dev_res = getResults(myImage, spots, device, Hlines, Vlines)
        del myImage

        _testAngles = testAngles(results, numRows, numCols)
    if not _testAngles:
        print("Auto-grid failed!")
        myImage = image.Image("/output/"+name+"/contrasted.bmp", copy_to_fb = True)
        myImage.to_grayscale(copy_to_fb=True)

        # PREPROCESSING
        k = preprocess(imgPath, myImage)
        myImage.save("/output/"+name+"/contrasted.bmp")
        memoryFlush()

        # GRIDDING
        Hlines, Vlines = gridding_v2(myImage, numRows, numCols)
        del myImage
        myImage = image.Image(imgPath, copy_to_fb = True)
        myImage.to_grayscale(copy_to_fb=True)
        memoryFlush()

        # SEGMENTATION
        spots = segmentation(myImage, Hlines, Vlines)

        # QUALITY
        results, controls, avg_res, std_dev_res = getResults(myImage, spots, device, Hlines, Vlines)
        del myImage

    totalPrint(name, results, controls, avg_res, std_dev_res, numRows, numCols, testAngles)

    memoryFlush()

