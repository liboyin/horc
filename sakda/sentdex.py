from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import Counter

def createSample():
    imageArraySample = open('sampleArray.txt','a')
    
    #Get sample only first 20 samples
    groupsWeHave = range(0,20)

    #Take only first 3 samples, left the last for test
    eachGroupHave = range(1,4)

    for eachGroup in groupsWeHave:
        for eachImage in eachGroupHave:
            imageFilePath = 'data/image' + str(eachGroup*5+eachImage).zfill(3) + '.JPG'
            sample = Image.open(imageFilePath)
            #sample = sample.convert('L')
            sample = np.asarray(sample)
            sample = str(sample.tolist())
            lineToWrite = str(eachGroup)+'::'+sample+'\n'
            imageArraySample.write(lineToWrite);

def whatImage(filePath):
    matchedAr = []
    loadSample = open('sampleArray.txt','r').read()
    loadSample = loadSample.split('\n')

    i = Image.open(filePath)
    i = np.asarray(i)
    i = i.tolist()

    inQuestion = str(i)
    for eachSample in loadSample:
        if len(eachSample) > 3:
            splitSample = eachSample.split('::')
            currentGroup = splitSample[0]
            currentImage = splitSample[1]
            eachPixSample = currentImage.split('],')
            eachPixInQ = inQuestion.split('],')
            x = 0;

            while x < len(eachPixSample):
                if eachPixSample[x] == eachPixInQ[x]:
                    matchedAr.append(int(currentGroup))
                x += 1

    x = Counter(matchedAr)
    print x
    
# This function used to populate sample array and save into a file.
# Once it exists, no need to call again.

#createSample()

#Test an image here
whatImage('data\image034.JPG')

'''
I1 = Image.open('data\image001.JPG')
I1 = I1.convert('L')
I1 = np.asarray(I1);
I1 = (I1 > I1.mean())*255
print I1

I2 = Image.open('data\image002.JPG')
I2 = I2.convert('L')
I2 = np.asarray(I2);
I2 = (I2 > I2.mean())*255

I3 = Image.open('data\image003.JPG')
I3 = I3.convert('L')
I3 = np.asarray(I3);
I3 = (I3 > I3.mean())*255

I4 = Image.open('data\image004.JPG')
I4 = I4.convert('L')
I4 = np.asarray(I4);
I4 = (I4 > I4.mean())*255

fig = plt.figure()
g1 = plt.subplot2grid((2,2), (0,0))
g2 = plt.subplot2grid((2,2), (0,1))
g3 = plt.subplot2grid((2,2), (1,0))
g4 = plt.subplot2grid((2,2), (1,1))

g1.imshow(I1,cmap='Greys')
g2.imshow(I2)
g3.imshow(I3)
g4.imshow(I4)
plt.show()
'''
