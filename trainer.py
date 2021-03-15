import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm_train
import pickle

def apply_features_svm(df):
    """ Apply the Steps Above for SVM"""
    # Reduce Forces below 25N to Zero
    for col in ['l1','l2','l3','l4','l5','l6','l7','l8','r1','r2','r3','r4','r5','r6','r7','r8']:
        df.loc[df[col] <= 25.0,col] = 0

    # Coefficient of Variation
    df['meanL'] = df.iloc[:,1:9].mean(axis = 1)
    df['meanR'] = df.iloc[:,9:17].mean(axis = 1)
    df['stdL'] = df.iloc[:,1:9].std(axis = 1)
    df['stdR'] = df.iloc[:,9:17].std(axis = 1)

    # Mean Center of Pressure
    lx = [-500,-700,-300,-700,-300,-700,-300,-500]
    rx = [500,700,300,700,300,700,300,500]
    ly = [-800,-400,-400,0,0,400,400,800]
    ry = [-800,-400,-400,0,0,400,400,800]

    for count, ele in enumerate(lx):
        df['slx' + str(count + 1)] = df['l' + str(count + 1)] * ele
    for count, ele in enumerate(ly):
        df['sly' + str(count + 1)] = df['l' + str(count + 1)] * ele
    for count, ele in enumerate(rx):
        df['srx' + str(count + 1)] = df['r' + str(count + 1)] * ele
    for count, ele in enumerate(ry):
        df['sry' + str(count + 1)] = df['r' + str(count + 1)] * ele

    df['cop_lx'] = df.iloc[:,24:32].sum(axis = 1)
    df['cop_ly'] = df.iloc[:,32:40].sum(axis = 1)
    df['cop_rx'] = df.iloc[:,40:48].sum(axis = 1)
    df['cop_ry'] = df.iloc[:,48:56].sum(axis = 1)

    for i in range(8):
        df = df.drop('slx' + str(i+1),axis = 1)
        df = df.drop('sly' + str(i+1),axis = 1)
        df = df.drop('srx' + str(i+1),axis = 1)
        df = df.drop('sry' + str(i+1),axis = 1)

    df['cop_lx'] = df['cop_lx'] / df['lTotal']
    df['cop_ly'] = df['cop_ly'] / df['lTotal']
    df['cop_rx'] = df['cop_rx'] / df['rTotal']
    df['cop_ry'] = df['cop_ry'] / df['rTotal']

    # Kurtosis
    df['kurt_l'] = df.iloc[:,1:9].kurtosis(axis = 1)
    df['kurt_r'] = df.iloc[:,9:17].kurtosis(axis = 1)

    # Skewness
    df['skew_l'] = df.iloc[:,1:9].skew(axis = 1)
    df['skew_r'] = df.iloc[:,9:17].skew(axis = 1)

    return df

def svm_predict(data):
    """Predcits Binary Class with Loaded SVM """
    df = pd.DataFrame(data)
    df = apply_features_svm(df)
    X = df
    print(str(X.shape))
    X = X.replace([np.inf,-np,inf],np.nan)
    X = X.fillna(X.mean())
    X = X.drop('time',axis = 1)

    # Check for np.isfinite(X).all()

    model = pickle.load(open('svm.sav','rb'))
    pred = model.predict(X)

    return pred