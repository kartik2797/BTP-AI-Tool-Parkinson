import pandas as pd
import numpy as np
%matplotlib inline 
import matplotlib.pyplot as plt
from os import listdir
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

forces_list = list()
files = listdir()
files.sort()
print(files)

def list_files(files):
  """ Creates the List of Files needed """
  to_delete = list()
  for file_ in files:
    if file_.find(".txt") != -1:
      if file_.find("_") != -1:
        pass
      else:
        to_delete.append(file_)
    else:
      to_delete.append(file_)
    
  for delete in to_delete:
    files.remove(delete)
    
  return files


files = list_files(files)
print(files)

def merge_time_data(files):
  """ Merging the time series data """
  ctrl_seqs = list()
  pd_seqs = list()
  count = 0
  for file_ in files:
    count += 1
    print(str(count) + " out of 306 Files done")
    if file_.find("Co") != -1:
      ctrl_seqs.append(file_)
    elif file_.find("Pt") != -1:
      pd_seqs.append(file_)
    else:
      print(file_)
  
  return ctrl_seqs,pd_seqs

ctrl_list,pk_list = merge_time_data(files)
print(ctrl_list)
print(pk_list)

df_wts = pd.read_csv('data_weights.csv',usecols = ['id','weight'])
df_wts.fillna(df_wts['weight'].mean(),inplace = True)
df_wts = df_wts.set_index('id')
df_wts.loc['GaCo01','weight']

np.random.seed(2)
from keras import layers
from keras import Input
from keras import optimizers
from keras.models import Model,Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, Dense,Conv1D
from keras.backend import l2_normalize
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
import datetime
from sklearn.metrics import confusion_matrix,  classification_report, accuracy_score
from scipy import stats
import pandas as pd
import random
import sys
from keras.utils import to_categorical
import keras

class Data_2D:
  """ For 2D Classification of the Data """
  def __init__(self):
      
      """ Initialize Variables """
      self.pk_list = pk_list
      self.ctrl_list = ctrl_list

      self.X_data = np.array([])
      self.y_data = np.array([])
      
      self.columns = ["time","l1","l2","l3","l4","l5","l6","l7","l8","r1","r2","r3","r4","r5","r6","r7","r8","lTotal","rTotal"]
      self.sep = "\t"

      self.df = pd.DataFrame()

      self.load()

  def load(self):
      """ Loads the Entire Data """

      print("Loading Training Control")
      self.load_data(self.ctrl_list, 0)
      print("Loading Training Parkinson")
      self.load_data(self.pk_list, 1)

      print("Shape of Entire DataFrame: " + str(self.df.shape))
      print(self.df.head())

  def load_data(self,liste,y):
      """ Loads the Respective Data """
      for i in range(0,len(liste)):
          
          df_unit = pd.read_csv(liste[i],sep = self.sep,names = self.columns)
          id_person = ((liste[i]).split('_'))[0]
          df_unit = df_unit.drop(['time'],axis = 1)
          df_unit = df_unit.div(df_wts.loc[id_person, 'weight'])

          df_unit['y'] = y

          if i == 0:
            print(df_unit.head())

          self.df = self.df.append(df_unit)


from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle
from sklearn import metrics
class model_svm:

  """ SVM Classifier """
  def __init__(self,df):
      """ Initialization """

      self.df = df
      self.train()

  def train(self):
      """ Training the Model """
      y = self.df['y']
      X = self.df.loc[:,self.df.columns != 'y']

      X = X.replace([np.inf,-np.inf],np.nan)
      X = X.fillna(X.mean())

      X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)
      
      clf = svm.SVC()
      print(X_train.shape)
      print(y_train.head())
      clf.fit(X_train,y_train)
      model_file = 'svm.sav'
      pickle.dump(model, open(model_file, 'wb'))
      y_pred = clf.predict(X_test)

      print("Shape of Y_pred: " + str(y_pred.shape))

      accuracy = self.accuracy(y_test,y_pred)

      print("Accuracy of the Model = " + str(accuracy))

  def accuracy(self,y_test,y_pred):
      """ Returning the Accuracy of the Model """
      return metrics.accuracy_score(y_test,y_pred)


data2D = Data_2D()
svm_ = model_svm(data2D.df)