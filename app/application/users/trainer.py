import numpy as np
import pandas as pd
from keras.models import load_model

def predict(name, weight, filename):
    """ The Main Function After Submit """
    
    cnn = load_model('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/testing/cnn_model')
    lstm = load_model('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/testing/lstm_model')
    svm = pickle.load(open('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/testing/svm.sav','rb'))
    dt = pickle.load(open('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/testing/dt.sav','rb'))

    df_raw = pd.read_csv('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/app/application/users/uploads',sep = "\t",columns = ["time","l1","l2","l3","l4","l5","l6","l7","l8","r1","r2","r3","r4","r5","r6","r7","r8","lTotal","rTotal"])
    df_raw = df_raw.drop(['time'],axis = 1)
    df_raw = df_raw.div(weight)

    X_data_3d = format_3d(df_raw)
    X_data_2d = format_2d(df_raw)

    pred_cnn = cnn_pred(X_data_3d,cnn)
    pred_lstm = lstm_pred(X_data_3d,lstm)
    pred_svm = svm_pred(X_data_2d,svm)
    pred_dt = dt_pred(X_data_2d,dt)

    pred_list = [cnn_pred,cnn_pred,cnn_pred,cnn_pred]
    print("Prediction List: " + str(pred_list))

    # Custom Model Training Also Left

    return pred_list

def format_3d(df):
    """ Formatting in 3D """
    X_data_raw = df.values
    X_data = np.array([])

    nb_datas = int(X_data_raw.shape[0] - 100)

    for start in range(0,nb_datas,50):
        end = start + 100
        X_temp = X_data_raw[start:end,:]

        if X_data.size == 0:
            X_data = X_temp
        else:
            X_data = np.dstack((X_data,X_temp))

    return X_data

def format_2d(df):
    """ Formatting in 2D """
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

        return df_formatdf_format = pd.DataFrame()

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

def cnn_pred(X_data,cnn):
    """ CNN Prediction """
    X_data = np.swapaxes(X_data,0,2)
    X_data = np.swapaxes(X_data,1,2)

    Y = cnn.predict(X_data)

    pred = np.mean(Y,axis = 0)
    pred = np.mean(pred,axis = 0)
    pred = np.around(pred)
    pred = int(pred)

    return pred

def lstm_pred(X_data,lstm):
    """ LSTM Prediction """
    X_data = np.swapaxes(X_data,0,2)
    X_data = np.swapaxes(X_data,1,2)

    Y = lstm.predict(X_data)
    pred = np.mean(Y,axis = 0)
    pred = np.around(pred)
    pred = int(pred)

    return pred

def svm_pred(X_data,svm):
    """ SVM Prediction """
    Y = svm.predict(X_data)
    pred = np.mean(Y,axis=0)
    pred = np.around(pred)
    pred = int(pred)

    return pred

def dt_pred(X_data,dt):
    """ DT Prediction """
    Y = self.dt.predict(X_data)
    pred = np.mean(Y,axis=0)
    pred = np.around(pred)
    pred = int(pred)

    return pred
