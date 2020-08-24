import cv2
import numpy as np
import os
import pickle
from scipy.misc import imresize
from hashkey import hashkey
from math import floor
from scipy import interpolate
from utils import gaussian2d


def cgls(A, b):
    height, width = A.shape
    x = np.zeros((height))
    while (True):
        sumA = A.sum()
        if (sumA < 100):
            break
        if (np.linalg.det(A) < 1):
            A = A + np.eye(height, width) * sumA * 0.000000005
        else:
            x = np.linalg.inv(A).dot(b)
            break
    return x


R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'trainData'

maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize / 2)
patchmargin = floor(patchsize / 2)
gradientmargin = floor(gradientsize / 2)

Q = np.zeros((Qangle, Qstrength, Qcoherence, R * R, patchsize * patchsize, patchsize * patchsize))
V = np.zeros((Qangle, Qstrength, Qcoherence, R * R, patchsize * patchsize))
h = np.zeros((Qangle, Qstrength, Qcoherence, R * R, patchsize * patchsize))
mark = np.zeros((Qstrength * Qcoherence, Qangle, R * R))

weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

image_list = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.dng')):
            image_list.append(os.path.join(parent, filename))

imagecount = 1
for image in image_list:
    print('\rProcessing image ' + str(imagecount) + ' of ' + str(len(image_list)) + ' (' + image + ')')
    origin = cv2.imread(image)

    grayorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min() / 255, grayorigin.max() / 255, cv2.NORM_MINMAX)

    height, width = grayorigin.shape
    LR = imresize(grayorigin, (floor((height + 1) / 2), floor((width + 1) / 2)), interp='bicubic', mode='F')

    height, width = LR.shape
    heightgrid = np.linspace(0, height - 1, height)
    widthgrid = np.linspace(0, width - 1, width)
    bilinearinterp = interpolate.interp2d(widthgrid, heightgrid, LR, kind='linear')
    heightgrid = np.linspace(0, height - 1, height * 2 - 1)
    widthgrid = np.linspace(0, width - 1, width * 2 - 1)
    upscaledLR = bilinearinterp(widthgrid, heightgrid)

    height, width = upscaledLR.shape
    operationcount = 0
    totaloperations = (height - 2 * margin) * (width - 2 * margin)
    for row in range(margin, height - margin):
        for col in range(margin, width - margin):
            if round(operationcount * 100 / totaloperations) != round((operationcount + 1) * 100 / totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount + 1) *  100 / totaloperations / 2), end='')
                print(' ' * (50 - round((operationcount + 1) * 100 / totaloperations / 2)), end='')
                print('|  ' + str(round((operationcount + 1) * 100 / totaloperations)) + '%', end='')

            operationcount += 1

            patch = upscaledLR[row - patchmargin:row + patchmargin + 1, col - patchmargin:col + patchmargin + 1]
            patch = np.matrix(patch.ravel())

            gradientblock = upscaledLR[row - gradientmargin:row + gradientmargin + 1, col - gradientmargin:col + gradientmargin + 1]

            angle, strength, coherence = hashkey(
                gradientblock, Qangle, weighting)

            pixeltype = ((row - margin) % R) * R + ((col - margin) % R)

            pixelHR = grayorigin[row, col]

            ATA = np.dot(patch.T, patch)
            ATb = np.dot(patch.T, pixelHR)
            ATb = np.array(ATb).ravel()

            Q[angle, strength, coherence, pixeltype] += ATA
            V[angle, strength, coherence, pixeltype] += ATb
            mark[coherence * 3 + strength, angle, pixeltype] += 1
    imagecount += 1

P = np.zeros((patchsize * patchsize, patchsize * patchsize, 7))
rotate = np.zeros((patchsize * patchsize, patchsize * patchsize))
flip = np.zeros((patchsize * patchsize, patchsize * patchsize))
for i in range(0, patchsize * patchsize):
    i1 = i % patchsize
    i2 = floor(i / patchsize)
    j = patchsize * patchsize - patchsize + i2 - patchsize * i1
    rotate[j, i] = 1
    k = patchsize * (i2 + 1) - i1 - 1
    flip[k, i] = 1
for i in range(1, 8):
    i1 = i % 4
    i2 = floor(i / 4)
    P[:, :, i - 1] = np.linalg.matrix_power(flip, i2).dot(np.linalg.matrix_power(rotate, i1))
Qextended = np.zeros((Qangle, Qstrength, Qcoherence, R * R, patchsize * patchsize, patchsize * patchsize))
Vextended = np.zeros((Qangle, Qstrength, Qcoherence, R * R, patchsize * patchsize))
for pixeltype in range(0, R * R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                for m in range(1, 8):
                    m1 = m % 4
                    m2 = floor(m / 4)
                    newangleslot = angle
                    if m2 == 1:
                        newangleslot = Qangle - angle - 1
                    newangleslot = int(newangleslot - Qangle / 2 * m1)
                    while newangleslot < 0:
                        newangleslot += Qangle
                    newQ = P[:, :, m - 1].T.dot(Q[angle, strength, coherence, pixeltype]).dot(P[:, :, m - 1])
                    newV = P[:, :, m - 1].T.dot(V[angle, strength, coherence, pixeltype])
                    Qextended[newangleslot, strength, coherence, pixeltype] += newQ
                    Vextended[newangleslot, strength, coherence, pixeltype] += newV
Q += Qextended
V += Vextended

print('Computing h ...')
operationcount = 0
totaloperations = R * R * Qangle * Qstrength * Qcoherence
for pixeltype in range(0, R * R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                operationcount += 1
                h[angle, strength, coherence, pixeltype] = cgls(
                    Q[angle, strength, coherence, pixeltype], V[angle, strength, coherence, pixeltype])

with open('./model/filter' + str(R) + 'x', 'wb') as fp:
    pickle.dump(h, fp)

print('\rFinished.')
