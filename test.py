import cv2
import numpy as np
import os
import pickle
from hashkey import hashkey
from math import floor
from scipy import interpolate
from utils import gaussian2d

R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'testData'

maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize / 2)
patchmargin = floor(patchsize / 2)
gradientmargin = floor(gradientsize / 2)

with open('./model/filter' + str(R) + 'x', 'rb') as fp:
    h = pickle.load(fp)

weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

imagecount = 1
for image in imagelist:
    print('\rUpscaling image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin = cv2.imread(image)

    ycrcvorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    grayorigin = ycrcvorigin[:, :, 0]

    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min() / 255, grayorigin.max() / 255, cv2.NORM_MINMAX)

    heightLR, widthLR = grayorigin.shape
    heightgridLR = np.linspace(0, heightLR - 1, heightLR)
    widthgridLR = np.linspace(0, widthLR - 1, widthLR)
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, grayorigin, kind='linear')
    heightgridHR = np.linspace(0, heightLR - 0.5, heightLR * 2)
    widthgridHR = np.linspace(0, widthLR - 0.5, widthLR * 2)
    upscaledLR = bilinearinterp(widthgridHR, heightgridHR)

    heightHR, widthHR = upscaledLR.shape
    predictHR = np.zeros((heightHR - 2 * margin, widthHR - 2 * margin))
    operationcount = 0
    totaloperations = (heightHR - 2 * margin) * (widthHR - 2 * margin)
    for row in range(margin, heightHR - margin):
        for col in range(margin, widthHR - margin):
            if round(operationcount * 100 / totaloperations) != round((operationcount + 1) * 100 / totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount + 1) * 100 / totaloperations / 2), end='')
                print(' ' * (50 - round((operationcount + 1) * 100 / totaloperations / 2)), end='')
                print('|  ' + str(round((operationcount + 1) * 100 / totaloperations)) + '%', end='')

            operationcount += 1

            patch = upscaledLR[row - patchmargin : row + patchmargin + 1, col - patchmargin : col + patchmargin + 1]
            patch = patch.ravel()

            gradientblock = upscaledLR[row - gradientmargin : row + gradientmargin + 1, col - gradientmargin : col + gradientmargin + 1]

            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)

            pixeltype = ((row - margin) % R) * R + ((col - margin) % R)
            predictHR[row - margin, col - margin] = patch.dot(h[angle, strength, coherence, pixeltype])

    predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)
    result = np.zeros((heightHR, widthHR, 3))
    y = ycrcvorigin[:, :, 0]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
    result[:, :, 0] = bilinearinterp(widthgridHR, heightgridHR)
    cr = ycrcvorigin[:, :, 1]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
    result[:, :, 1] = bilinearinterp(widthgridHR, heightgridHR)
    cv = ycrcvorigin[:, :, 2]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
    result[:, :, 2] = bilinearinterp(widthgridHR, heightgridHR)
    result[margin : heightHR - margin, margin : widthHR - margin, 0] = predictHR
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    cv2.imwrite('output/' + os.path.splitext(os.path.basename(image))[0] + '_' + str(R) + 'x_result.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    imagecount += 1

print('\rFinished.')
