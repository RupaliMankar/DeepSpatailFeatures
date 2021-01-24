from dataload import  DataGeneratorAutoCNN
from dataload import  DataGeneratorAutoPredictCNN
from dataPrepare import loaddataCNN
from dataPrepare import compileimages
from tensorflow.keras import backend as K
from models import autounet
import matplotlib.pyplot as plt
from tensorflow import keras

#-------------------main script---------------------------------------------------------------------
maskpath = "T:/Rupali/pansharpening/OVARY"  #maskpath 
ftirpath = "T:/Rupali/pansharpening/OVARY" #EnvifilePath

tiles = ["c2","c4"] # list of cores
#maskfiles = ["#/cl-epith.bmp", "#/cl-stroma.bmp"] #list of classes for raw
maskfiles = ["#/cl-epith.bmp", "#/cl-stroma.bmp"] #list of classes for raw
ftirfile =  "#/rowc-b-#-up-select-n-bip" #raw
darkfile = "#/#_dark_aligned.png"
savefile = "T:/Rupali/pansharpening/cnn/OVARY/Unet-4X4X16-im64_#.h5" #save results in this file

imsize = 64
num_categories = len(maskfiles) #number of categories means number of classes
sz = imsize*2


X_train_loc, y_train = loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath)
images = compileimages(tiles, ftirpath, darkfile)

dg = DataGeneratorAutoCNN(images, X_train_loc, imsize, batch_size=128)
#dg = DataGeneratorAutoCNN(images, X_train_loc, y_train, imsize, num_categories, batch_size=128)


#autoencoder = buildmodel(1, imsize)
autoencoder = autounet()
#autoencoder = unet(imsize, num_categories)
for i in range(10):   
    #train a model
    autoencoder.fit_generator(dg, epochs = 5, verbose=1)
    autoencoder.save(savefile.replace('#',str(i))) #save a model


# #validate model
tiles = ["c4"] 
X_val_loc, y_val = loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath)
images = compileimages(tiles, ftirpath, darkfile)
#dgpredict = DataGeneratorAutoPredictCNN(images, X_val_loc, y_val, imsize, batch_size=128, random=False)
dgpredict = DataGeneratorAutoPredictCNN(images, X_val_loc, imsize, batch_size=128, random=False)
predprob = autoencoder.predict_generator(dgpredict, verbose=1)

P = dgpredict.__getitem__(4)  #get data for ith batch
plt.imshow(P[13,:,:,0].reshape(sz,sz))
plt.imshow(predprob[525,:,:,0].reshape(sz,sz))
P = dgpredict.__getitem__(937)  #get data for ith batch
plt.imshow(P[64,:,:,0].reshape(sz,sz))
plt.imshow(predprob[120000,:,:,0].reshape(sz,sz))

# #fetching model infromation
for l in autoencoder.layers:
    print(l.name)

autoencoder = keras.models.load_model(savefile)#load pretrained model
#method for extracting featurs
feature_extractor = keras.Model(
    inputs=autoencoder.inputs,
    outputs=[autoencoder.get_layer('conv2d_10').output]
    #outputs = [layer.output for layer in autoencoder.layers]
)

tiles = ["c8"] 
maskfiles = ["#/mask-up.bmp"] #list of classes for raw

X_val_loc, y_val = loaddataCNN(tiles, maskfiles, ftirfile, maskpath, ftirpath,)
images = compileimages(tiles, ftirpath, darkfile)
dgpredict = DataGeneratorAutoPredictCNN(images, X_val_loc, imsize, batch_size=128, random=False)
features = feature_extractor.predict_generator(dgpredict, verbose=1)



#features = feature_extractor(P[10,:,:,:]) #extract features for one example
for i in range(4):
    K.clear_session()
 # load model
#model = keras.models.load_model(savefile)
# summarize model.
#model.summary()