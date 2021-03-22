import numpy as np
import pandas as pd
from keras.models import load_model

data = np.loadtxt('sample.txt')
data = data[:,np.arange(1,19)]
data = data // 70

print(str(data.shape))

model = load_model('/home/sparsh/BTP/BTP-AI-Tool-Parkinson/Saved_Models/CNN')


X = np.array([])
nb_datas = int(data.shape[0] - 100)
for start in range(0,nb_datas,50):
    end = start + 100
    temp = data[start:end,:]
    if X.size == 0:
        X = temp
    else:
        X = np.dstack((X,temp))

X = np.swapaxes(X,0,2)
X = np.swapaxes(X,1,2)

Y = model.predict(np.split(X, X.shape[2], axis=2))
print(Y)

pred = np.mean(Y, axis = 0)
pred = np.around(pred)

pred = int(pred)
print(str(pred))