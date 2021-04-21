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
datas = np.loadtxt('sample.txt')
	datas = datas[:,np.arange(1,19)]
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
datas = np.delete(datas,num,1)
