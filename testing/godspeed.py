import numpy as np
import pandas as pd
from keras.models import load_model
import pickle
from os import listdir
import os

####################### LOADING THE FILES ############################
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
##########################################################################


class godspeed:
    """ Combined Accuracy Classfier """

    def __init__(self,pk_list,ctrl_list):
        """ Initialization """

        self.pk_list = pk_list
        self.ctrl_list = ctrl_list

        self.gait_cycle = 100
        self.steps = 50
        self.columns = ["time","l1","l2","l3","l4","l5","l6","l7","l8","r1","r2","r3","r4","r5","r6","r7","r8","lTotal","rTotal"]
        self.sep = "\t"

        self.cnn = load_model('cnn_model/')
        self.lstm = load_model('lstm_model/')

        self.svm = pickle.load(open('svm.sav','rb'))
        self.dt = pickle.load(open('dt.sav','rb'))

        self.df = pd.DataFrame()

        self.accumulate()

    def accumulate(self):
        """ The Main Training Method """

        count = 0
        for fname in self.pk_list:
            
            pred_cnn = self.cnn_pred(fname)
            pred_lstm = self.lstm_pred(fname)
            pred_svm = self.svm_pred(fname)
            pred_dt = self.dt_pred(fname)

            # Getting the Predictions in 0 and 1

            pred_real = 1 # this is pk_list for Parkinson
            
            self.df.loc[count,"File"] = fname
            self.df.loc[count, "pred_cnn"] = pred_cnn
            self.df.loc[count, "pred_lstm"] = pred_lstm
            self.df.loc[count, "pred_svm"] = pred_svm
            self.df.loc[count, "pred_dt"] = pred_dt
            self.df.loc[count, "pred_real"] = 1

            count+=1
            print(str(count) + " Files out of " + str(len(pk_list)) + " Done")

        print("Completed Parkison List")

        count = 0
        for fname in self.ctrl_list:
            
            pred_cnn = self.cnn_pred(fname)
            pred_lstm = self.lstm_pred(fname)
            pred_svm = self.svm_pred(fname)
            pred_dt = self.dt_pred(fname)

            # Getting the Predictions in 0 and 1

            pred_real = 0 # this is pk_list for Parkinson
            
            self.df.loc[count,"File"] = fname
            self.df.loc[count, "pred_cnn"] = pred_cnn
            self.df.loc[count, "pred_lstm"] = pred_lstm
            self.df.loc[count, "pred_svm"] = pred_svm
            self.df.loc[count, "pred_dt"] = pred_dt
            self.df.loc[count, "pred_real"] = 0

            count+=1
            print(str(count) + " Files out of " + str(len(ctrl_list)) + " Done")

        print("Completed Control List")

        print("Shape of Entire DataFrame is : " + str(self.df.shape))
        print("Columns of DataFrame is : " + str(self.df.columns))
        print("Head of DataFrame is : " + str(self.df.head(10)))

        self.train()

    def train(self):
        """ Training the LR Model """
        print("Wait...")

    def cnn_pred(self,fname):
        """ Prediction of CNN """
        X_data = self.format_data_3d(fname)
        X_data = np.swapaxes(X_data,0,2)
        X_data = np.swapaxes(X_data,1,2)

        print("Shape of X_data in 3D: " + str(X_data.shape) + " for File:- " + fname)

        Y = self.cnn.predict(X_data)
        pred = np.mean(Y,axis = 0)
        pred = np.around(pred)
        pred = int(pred)

        print("Value by CNN for " + fname + " is " + str(pred))
        return pred
    
    def lstm_pred(self,fname):
        """ Prediction of LSTM """
        X_data = self.format_data_3d(fname)
        X_data = np.swapaxes(X_data,0,2)
        X_data = np.swapaxes(X_data,1,2)

        Y = self.lstm.predict(X_data)
        pred = np.mean(Y,axis = 0)
        pred = np.around(pred)
        pred = int(pred)

        print("Value by LSTM for " + fname + " is " + str(pred))
        return pred

    def svm_pred(self,fname):
        """ Prediction of SVM """
        X_data = self.format_data_2d(fname)

        print("Shape of X_data in 2D: " + str(X_data.shape) + " for File:- " + fname)

        Y = self.svm.predict(X_data)
        print("Only for SVM [IGNORE]: " + str(Y))
        pred = np.mean(Y,axis=0)
        pred = np.around(pred)
        pred = int(pred)

        print("Value by SVM for " + fname + " is " + str(pred))
        return pred

    def dt_pred(self,fname):
        """ Prediction of Decision Tree """
        X_data = self.format_data_2d(fname)

        print("Shape of X_data in 2D: " + str(X_data.shape) + " for File:- " + fname)

        Y = self.dt.predict(X_data)
        pred = np.mean(Y,axis=0)
        pred = np.around(pred)
        pred = int(pred)

        print("Value by Decision Tree for " + fname + " is " + str(pred))
        return pred
        

    def format_data_3d(self,fname):
        """ Formating Data for 3D Models """
        df = pd.read_csv(fname,sep = self.sep,names = self.columns)
        id_person = (fname).split('_')[0]
        df = df.drop(['time'],axis = 1)
        df = df.div(df_wts.loc[id_person,'weight'])

        X_data_raw = df.values
        X_data = np.array([])

        nb_datas = int(X_data_raw.shape[0] - self.gait_cycle)

        for start in range(0,nb_datas,self.steps):
            end = start + self.gait_cycle
            X_temp = X_data_raw[start:end,:]

            if X_data.size == 0:
                X_data = X_temp
            else:
                X_data = np.dstack((X_data,X_temp))

        return X_data
    
    def format_data_2d(self,fname):
        """ Formatting Data for 2D Models """
        df = pd.read_csv(fname,sep = self.sep,names = self.columns)
        id_person = (fname).split('_')[0]
        df = df.drop(['time'],axis = 1)
        df = df.div(df_wts.loc[id_person,'weight'])

        df_format = pd.DataFrame()

        for col in df.columns:
            for x in ["Min", "Max", "Std", "Med", "Avg", "Skewness", "Kurtosis"]:
                colname = col + x
                if x == "Min":
                    df_format.loc[0,colname] = df[col].min(axis = 0)

                if x == "Max":
                    df_format.loc[0,colname] = df[col].max(axis = 0)

                if x == "Std":
                    df_format.loc[0,colname] = df[col].std(axis = 0)
                
                if x == "Med":
                    df_format.loc[0,colname] = df[col].median(axis = 0)
                
                if x == "Avg":
                    df_format.loc[0,colname] = df[col].mean(axis = 0)
                
                if x == "Skewness":
                    df_format.loc[0,colname] = df[col].skew(axis = 0)
                    
                if x == "Kurtosis":
                    df_format.loc[0,colname] = df[col].skew(axis = 0)

        df_format = df_format.replace([np.inf,-np.inf],np.nan)
        df_format = df_format.fillna(df_format.mean())

        return df_format
                

god = godspeed(pk_list,ctrl_list)
print("Shape of God is: " + str(god.df.shape))