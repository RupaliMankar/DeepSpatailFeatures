import os
import numpy as np
import envi
import matplotlib.pyplot as plt


def loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath, samples=None):
    os.chdir(ftirpath)
    ftirenvifile = envi.envi(ftirfile.replace("#", tiles[0]))
    dataXloc = np.empty((0, 3),dtype=np.int)
    dataY = np.empty((0),dtype=np.int)
    for j in range(len(tiles)):
        os.chdir(ftirpath)
        ftirenvifile = envi.envi(ftirfile.replace("#", str(tiles[j])))
        maskarray = np.empty((0, ftirenvifile.header.lines, ftirenvifile.header.samples))
        for maskfile in maskfiles:
            os.chdir(maskpath)
            mask = plt.imread(maskfile.replace("#", str(tiles[j])))
            maskarray = np.append(maskarray, np.reshape(mask[:,:,0], (1, mask.shape[0], mask.shape[1])), axis=0)
        datalocations = np.transpose(np.nonzero(maskarray))
        dataXseg = np.empty((len(datalocations),3),dtype=np.int)
        dataYseg = np.empty((len(datalocations)),dtype=np.int)
        for i in range(int(len(datalocations))):
            location = datalocations[i]
            dataYseg[i] = location[0] #dataYseg[i] = datalocations[i][0]
            dataXseg[i,:] = [j,location[1],location[2]] #[j,datalocations[i][1:3]]
            if i % 25 == 0:
                print("\r" + str(int(float(i) / len(datalocations) * 100)) + "% loaded in tile " + str(tiles[j]), end="")
        print("\n")
        dataXloc = np.append(dataXloc,dataXseg,axis=0)  #append data locations for each core
        dataY = np.append(dataY,dataYseg,axis=0)
    if samples != None: #randomly select given number of samples from each class
        newdataXloc = np.empty((samples*len(maskfiles),3),dtype=np.int)     #1. CoreLabel 2. [Y X] 
        newdataY = np.empty((samples*len(maskfiles)),dtype=np.int)    #classLabel
        index_of_first_class_sample = 0
        for k in range(len(maskfiles)):  #for each class
            locs = np.argwhere(dataY == k) #find locations of these class k pixels
            np.random.shuffle(locs)
            num_of_samples = samples 
            if len(locs)<samples:
                num_of_samples = len(locs) 
            
            index_of_last_class_sample = index_of_first_class_sample + num_of_samples
           
            newdataY[index_of_first_class_sample:index_of_last_class_sample] = np.full((num_of_samples), k)  #fill Y with class labels of class k
            newdataXloc[index_of_first_class_sample:index_of_last_class_sample, :] = np.reshape(dataXloc[locs[0:num_of_samples],:], (num_of_samples,3))
            
            index_of_first_class_sample = index_of_first_class_sample + num_of_samples
        dataXloc = newdataXloc[:index_of_first_class_sample,:]
        dataY = newdataY[:index_of_first_class_sample]

    return dataXloc, dataY

def compileimages(tiles, ftirpath, darkfile):
    os.chdir(ftirpath)
    imcoord = []
    imagesarr = np.empty((0,1))
    for tile in tiles:
        dark_img = plt.imread(darkfile.replace("#", str(tile)))
        imcoord.append([dark_img.shape[0], dark_img.shape[1]]) #Y = 1 X = 2 check X and Y dimension
        #dark_img = np.moveaxis(dark_img,0,-1) #envi file is Y*X*B
        imagesarr = np.append(imagesarr,dark_img.reshape((dark_img.shape[0]*dark_img.shape[1],1)),axis=0) #2d concatenated matrix for all cores (Y*X)*B
    #images = np.reshape(images,(len(tiles),ftirenvifile.header.bands,ftirenvifile.header.lines*ftirenvifile.header.samples))
    #images = np.moveaxis(images,1,-1)
    #images = np.reshape(images,(len(tiles)*ftirenvifile.header.lines*ftirenvifile.header.samples,ftirenvifile.header.bands)))
    images = []
    idx = 0
    for coordinates in imcoord:
        #Take (Y*X)*B and convert to B*(Y*X) then reshape it to B*Y*X
        images.append(np.reshape(np.moveaxis(imagesarr[idx:idx+coordinates[0]*coordinates[1],:],-1,0),(imagesarr.shape[-1],coordinates[0],coordinates[1])))
        idx = idx + coordinates[0]*coordinates[1]
    return images