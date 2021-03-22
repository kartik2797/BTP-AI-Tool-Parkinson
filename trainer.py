import numpy as np
import pandas as pd
from ml_models.cnn import CNN
def predict(name, weight, filename):
    """ The Main Function After Submit """
    
    # CNN Model
    cnn_pred = CNN(weight = weight,filename = filename)
    
    pred_list = [cnn_pred,cnn_pred,cnn_pred,cnn_pred]

    # Custom Model Training Also Left

    return pred_list

    