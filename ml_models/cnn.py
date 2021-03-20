import numpy as np
import pandas as pd
from keras.models import load_model
class CNN:

    def __init__(self,weight,filename,features = np.arange(1,19)):
        """ Sets the Init Values for the Class """
        self.features_to_load = features
        self.nb_features = self.features_to_load.shape[0]
        self.weight = weight
        self.filename = filename
        self.gait_cycle = 100
        self.X_data = self.load()
        self.pred = self.predict()

    def load(self):
        """ Loads the Dataset """
        data = np.loadtxt('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/testing/sample.txt') # Instead of FileName it should be File Location 
        print(str(data.shape))
        data = data[:,np.arange(1,19)]
        # data = np.true_divide(data, self.weight)
        X = np.array([])
        nb_datas = int(data.shape[0] - self.gait_cycle)
        for start in range(0,nb_datas,50):
            end = start + 100
            temp = data[start:end,:]
            if X.size == 0:
                X = temp
            else:
                X = np.dstack((X,temp))
        
        X = np.swapaxes(X,0,2)
        X = np.swapaxes(X,1,2)

        return X
    
    def predict(self):
        """ Prediction """
        model = load_model('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/Saved_Models/CNN')
        Y = model.predict(np.split(self.X_data, self.X_data.shape[2], axis=2))
        pred = np.mean(Y, axis = 0)
        pred = np.around(pred)
        pred = int(pred)

        return pred


    




    