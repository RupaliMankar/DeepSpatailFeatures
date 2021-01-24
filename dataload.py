from tensorflow import keras
import numpy as np
import math
import os

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_array, label_dict, folder_path, data_size, num_categories, batch_size=64):
        self.data_array = data_array
        self.label_dict = label_dict
        self.folder_path = folder_path
        self.data_size = data_size
        self.num_categories = num_categories
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        """
        Returns number of batches in epoch
        """
        return math.ceil(len(self.data_array) / self.batch_size)

    def  __getitem__(self, batch):
        """
        Called to get next batch of data
        """
        batch_index = self.index_list[batch*self.batch_size:batch*self.batch_size + self.__get_batch_size(batch)]
        batch_data = [self.data_array[i] for i in batch_index]
        X, y = self.__generate_data(batch_data)
        return X, y

    def __get_batch_size(self, batch):
        """
        Returns the size of the batch
        """
        if batch == len(self)-1:
            return len(self.data_array) - self.batch_size*(batch-1)
        else:
            return self.batch_size

    def __generate_data(self, batch_data):
        """
        Compiles data array sized for a batch
        """
        X = np.empty((len(batch_data), *self.data_size))
        y = np.empty((len(batch_data)))
        for j in range(len(batch_data)):
            X[j,] = np.load(os.path.join(self.folder_path, batch_data[j] + ".npy"))
            y[j] = self.label_dict[batch_data[j]]
        return(X, keras.utils.to_categorical(y, num_classes=self.num_categories))


    def on_epoch_end(self):
        """
        Resets data paths after epoch end
        """
        self.index_list = np.arange(len(self.data_array))
        np.random.shuffle(self.index_list)


class DataGeneratorCNN(keras.utils.Sequence):

    def __init__(self, images, data_array_locations, label_dict, imsize, num_categories, batch_size=64, random=True):
        self.images = images
        self.data_array_locations = data_array_locations
        self.imsize = imsize
        self.label_dict = label_dict
        self.num_categories = num_categories
        self.batch_size = batch_size
        self.random = random
        self.on_epoch_end()

    def __len__(self):
        """
        Returns number of batches in epoch
        """
        return math.ceil(len(self.data_array_locations) / self.batch_size)

    def  __getitem__(self, batch):
        """
        Called to get next batch of data
        """
        batch_index = self.index_list[batch*self.batch_size:batch*self.batch_size + self.__get_batch_size(batch)]
        batch_data = [self.data_array_locations[i,] for i in batch_index]
        X, y = self.__generate_data(batch_index, batch_data)
        return X, y

    def __get_batch_size(self, batch):
        """
        Returns the size of the batch
        """
        if batch == len(self)-1:
            return len(self.data_array_locations) - self.batch_size*(batch-1)
        else:
            return self.batch_size

    def __generate_data(self, batch_index, batch_data):
        """
        Compiles data array sized for a batch
        """
        X = np.zeros((len(batch_data), 2*self.imsize, 2*self.imsize, self.images[0].shape[0]))
        y = np.zeros((len(batch_data)))
        for j in range(len(batch_data)):
            #if np.moveaxis(self.images[batch_data[j][0]][:,batch_data[j][1]-self.imsize:batch_data[j][1]+self.imsize+1,batch_data[j][2]-self.imsize:batch_data[j][2]+self.imsize+1],0,-1).shape == (1+2*self.imsize,1+2*self.imsize,self.images[0].shape[0]):
            try:
                X[j,] = np.moveaxis(self.images[batch_data[j][0]][:,batch_data[j][1]-self.imsize:batch_data[j][1]+self.imsize+1,batch_data[j][2]-self.imsize:batch_data[j][2]+self.imsize+1],0,-1)
            except:
                pass
            y[j] = self.label_dict[batch_index[j]]
        return(X, keras.utils.to_categorical(y, num_classes=self.num_categories))


    def on_epoch_end(self):
        """
        Resets data paths after epoch end
        """
        self.index_list = np.arange(self.data_array_locations.shape[0])
        if self.random:
            np.random.shuffle(self.index_list)



class DataGeneratorAutoCNN(keras.utils.Sequence):

    def __init__(self, images, data_array_locations, imsize, batch_size=64, random=True):
        self.images = images
        self.data_array_locations = data_array_locations
        self.imsize = imsize
        self.batch_size = batch_size
        self.random = random
        self.on_epoch_end()

    def __len__(self):
        """
        Returns number of batches in epoch
        """
        return math.ceil(len(self.data_array_locations) / self.batch_size)

    def  __getitem__(self, batch):
        """
        Called to get next batch of data
        """
        batch_index = self.index_list[batch*self.batch_size:batch*self.batch_size + self.__get_batch_size(batch)]
        batch_data = [self.data_array_locations[i,] for i in batch_index]
        X = self.__generate_data(batch_index, batch_data)
        return X, X

    def __get_batch_size(self, batch):
        """
        Returns the size of the batch
        """
        if batch == len(self)-1:
            return len(self.data_array_locations) - self.batch_size*(batch-1)
        else:
            return self.batch_size

    def __generate_data(self, batch_index, batch_data):
        """
        Compiles data array sized for a batch
        """
        X = np.zeros((len(batch_data), 2*self.imsize, 2*self.imsize, self.images[0].shape[0]))
        for j in range(len(batch_data)):
            #if np.moveaxis(self.images[batch_data[j][0]][:,batch_data[j][1]-self.imsize:batch_data[j][1]+self.imsize+1,batch_data[j][2]-self.imsize:batch_data[j][2]+self.imsize+1],0,-1).shape == (1+2*self.imsize,1+2*self.imsize,self.images[0].shape[0]):
            core_ind = batch_data[j][0]
            pix_ind = batch_data[j][1:3]
            try:
               # X[j,] = np.moveaxis(self.images[batch_data[j][0]][:,batch_data[j][1]-self.imsize:batch_data[j][1]+self.imsize+1,batch_data[j][2]-self.imsize:batch_data[j][2]+self.imsize+1], 0, -1)
                X[j,] = np.moveaxis(self.images[core_ind][:, pix_ind[0]-self.imsize:pix_ind[0]+self.imsize, pix_ind[1]-self.imsize:pix_ind[1]+self.imsize], 0, -1)
            except:
                pass
           
        return(X)


    def on_epoch_end(self):
        """
        Resets data paths after epoch end
        """
        self.index_list = np.arange(self.data_array_locations.shape[0])
        if self.random:
            np.random.shuffle(self.index_list)


class DataGeneratorAutoPredictCNN(keras.utils.Sequence):

    def __init__(self, images, data_array_locations, imsize, batch_size=64, random=True):
        self.images = images
        self.data_array_locations = data_array_locations
        self.imsize = imsize
        self.batch_size = batch_size
        self.random = random
        self.on_epoch_end()

    def __len__(self):
        """
        Returns number of batches in epoch
        """
        return math.ceil(len(self.data_array_locations) / self.batch_size)

    def  __getitem__(self, batch):
        """
        Called to get next batch of data
        """
        batch_index = self.index_list[batch*self.batch_size:batch*self.batch_size + self.__get_batch_size(batch)]
        batch_data = [self.data_array_locations[i,] for i in batch_index]
        X = self.__generate_data(batch_index, batch_data)
        return X

    def __get_batch_size(self, batch):
        """
        Returns the size of the batch
        """
        if batch == len(self)-1:
            return len(self.data_array_locations) - self.batch_size*(batch-1)
        else:
            return self.batch_size

    def __generate_data(self, batch_index, batch_data):
        """
        Compiles data array sized for a batch
        """
        X = np.zeros((len(batch_data), 2*self.imsize, 2*self.imsize, self.images[0].shape[0]))
        for j in range(len(batch_data)):
            #if np.moveaxis(self.images[batch_data[j][0]][:,batch_data[j][1]-self.imsize:batch_data[j][1]+self.imsize+1,batch_data[j][2]-self.imsize:batch_data[j][2]+self.imsize+1],0,-1).shape == (1+2*self.imsize,1+2*self.imsize,self.images[0].shape[0]):
            core_ind = batch_data[j][0]
            pix_ind = batch_data[j][1:3]
            try:
               # X[j,] = np.moveaxis(self.images[batch_data[j][0]][:,batch_data[j][1]-self.imsize:batch_data[j][1]+self.imsize+1,batch_data[j][2]-self.imsize:batch_data[j][2]+self.imsize+1], 0, -1)
                X[j,] = np.moveaxis(self.images[core_ind][:, pix_ind[0]-self.imsize:pix_ind[0]+self.imsize, pix_ind[1]-self.imsize:pix_ind[1]+self.imsize], 0, -1)
            except:
                pass
           
        return(X)


    def on_epoch_end(self):
        """
        Resets data paths after epoch end
        """
        self.index_list = np.arange(self.data_array_locations.shape[0])
        if self.random:
            np.random.shuffle(self.index_list)
