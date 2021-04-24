import numpy as np
import pandas as pd
from keras import Input
from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Dense,LSTM,Dropout, Flatten, Convolution2D, MaxPooling2D, Dense,Conv1D
from keras.backend import l2_normalize
import os
from keras.utils import to_categorical
import random
import sys

class Data:
  """ For Data Loading """

  def __init__(self, steps = 50, features = np.arange(1,19),gait_cycle = 100):
    """ Initialization """

    self.features_to_load = np.arange(1,19)
    self.gait_cycle = gait_cycle
    self.steps = steps

    self.pk_list = pk_list
    self.ctrl_list = ctrl_list

    self.X_data = np.array([])
    self.y_data = np.array([])

    self.columns = ["time","l1","l2","l3","l4","l5","l6","l7","l8","r1","r2","r3","r4","r5","r6","r7","r8","lTotal","rTotal"]
    self.sep = "\t"


    self.load()

  def load(self):
      """ Loads the Data in CSV """
      print("Loading Control Data")
      self.load_data(self.ctrl_list,0)
      print("Loading Parkinson Data")
      self.load_data(self.pk_list,1)

  def load_data(self,liste,y):
      """ Loads the data in CSV of one list """

      for i in range(0,len(liste)):
          df_unit = pd.read_csv(liste[i],sep = self.sep,names = self.columns)
          id_person = ((liste[i]).split('_'))[0]
          df_unit = df_unit.drop(['time'],axis = 1)
          df_unit = df_unit.div(df_wts.loc[id_person, 'weight'])

          print(df_unit.shape[0])

          df_preprocess_unit = self.preprocess(df_unit)
          X_data,y_data = self.generate_data(df_preprocess_unit,y)

          if self.X_data.size == 0:
              self.X_data = X_data
              self.y_data = y_data
          else:
              self.X_data = np.dstack((self.X_data,X_data))
              self.y_data = np.vstack((self.y_data,y_data))

          print(X_data.shape, self.X_data.shape)
          print("Printing Shape of Y")
          print(y_data.shape, self.y_data.shape)



  def generate_data(self,df_unit,y):
      """ Generates the data into right formats """
      X_data_unsampled = df_unit.values
      X_data = np.array([])
      y_data = np.array([])
      
      nb_datas = int(X_data_unsampled.shape[0] - self.gait_cycle)
      
      for start in range(0,nb_datas,self.steps):
          end = start + self.gait_cycle
          X_temp = X_data_unsampled[start:end,:]
          
          if X_data.size == 0:
              X_data = X_temp
              y_data = y
          else:
              X_data = np.dstack((X_data, X_temp))
              y_data = np.vstack((y_data,y))

      return X_data,y_data
            

  def preprocess(self,df_unit):
      """ Preprocessing Steps """
      return df_unit

from keras.layers import Conv2D,Activation,MaxPooling1D
class model_cnn:
  """ CNN Model """

  def __init__(self,X_data,y_data):
      """ Initialization """
      self.X_data = X_data
      self.y_data = y_data
      self.X_train = np.array([])
      self.y_train = np.array([])
      self.X_test = np.array([])
      self.y_test = np.array([])

      self.model_train()
  
  def model_train(self):
      """ Training the Model """
      self.train_test_split(split = 0.8)

      model = Sequential()
      model.add(Conv1D(32,5,padding = "same",activation="relu",input_shape = (100,18)))
      model.add(MaxPooling1D(2))
      model.add(Conv1D(64,5,padding = "same",activation="relu"))
      model.add(MaxPooling1D(2))
      model.add(Dropout((0.25)))
      model.add(Dense(1024))
      model.add(Activation("softmax"))

      model.add(Dense(1))

      print(model.summary())

      print("Compiling...")

      model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics = ['accuracy'])
      history = model.fit(self.X_train, \
                          self.y_train, \
                          verbose = 1, \
                          shuffle = True, \
                          epochs = 100, \
                          batch_size = 512, \
                          validation_data = (self.X_test,self.y_test))

      model.save("cnn_model")

      self.predict(model)

  def train_test_split(self,split):
    """ Splitting the Data """

    train = int(0.8 * self.X_data.shape[2])
    
    self.X_train = self.X_data[:,:,:train]
    self.y_train = self.y_data[:train,:]
    self.X_test = self.X_data[:,:,train:]
    self.y_test = self.y_data[train:,:]

    self.X_train = np.swapaxes(self.X_train,0,2)
    self.X_train = np.swapaxes(self.X_train,1,2)
    self.X_test = np.swapaxes(self.X_test,0,2)
    self.X_test = np.swapaxes(self.X_test,1,2)

    print("Shape of Training Set: " + str(self.X_train.shape))
    print("Shape of Training Set (Y): " + str(self.y_train.shape))
    print("Shape of Test Set: " + str(self.X_test.shape))

  def predict(self,model):
    """ Evaluating the Network """
    predictions = model.predict(x = X_test,batch_size = 512)
    print(classification_report(y_test.argmax(axis = 1),predictions.argmax(axis = 1),target_names = [0,1]))


datas = np.loadtxt('sample.txt')
datas = datas[:,np.arange(1,19)]

datas = Data(50,features = np.arange(1,19),gait_cycle=100)
cnn = model_cnn(data.X_data,data.y_data)