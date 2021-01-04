import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_iris
from sklearn import preprocessing

ree = 0
getter1 = pandas.read_csv('testing_nfl.csv')
getter2 = getter1.values
train = np.zeros([1, getter2.shape[1]])
for i in range(len(getter2)):
    try:
        if not getter2[i, 3] == '-' and not math.isnan(float(getter2[i, 3])):
            train = np.vstack((train, getter2[i, :]))
    except:
        pass

X_trn_raw, y_trn = train[1:,3:19], train[1:,21]
y_trn = np.array([float(item) for item in y_trn])
y_trn = y_trn.astype(np.float32)
X_trn_raw = np.array([[float(elm) for elm in row] for row in X_trn_raw])
X_trn_raw = X_trn_raw.astype(np.float32)
for i in range(len(X_trn_raw)):
    X_trn_raw[i, 1] = X_trn_raw[i, 0] - X_trn_raw[i, 1]
    X_trn_raw[i, 9] = X_trn_raw[i, 8] - X_trn_raw[i, 9]

pandas.DataFrame(X_trn_raw).to_csv("raw.csv")

inputs = X_trn_raw - np.mean(X_trn_raw,axis=0) #shift
inputs = inputs/(np.max(inputs,axis=0)) #normalize
inputs = np.concatenate((inputs, np.ones((X_trn_raw.shape[0],1))), axis = 1) #add bias

inputs_new = np.concatenate((X_trn_raw[:,0:8],np.square(X_trn_raw[:,0:8])), axis=1) #add square term
inputs_new = inputs_new - np.mean(inputs_new,axis=0) #shift
inputs_new = inputs_new/(np.max(inputs_new,axis=0)) #normalize
inputs_new_bruteforce = np.zeros(inputs_new.shape)

while ree < len(X_trn_raw):
    inputs_new_bruteforce[ree, :] = inputs_new[ree+1, :]
    inputs_new_bruteforce[ree+1, :] = inputs_new[ree, :]
    inputs[ree, 8:16] = inputs[ree+1, 0:8]
    inputs[ree+1, 8:16] = inputs[ree, 0:8]
    ree += 2

pandas.DataFrame(inputs_new).to_csv("junk.csv")

#print(inputs_new_bruteforce)

inputs_new = np.concatenate((inputs_new, inputs_new_bruteforce), axis=1)
inputs_new = np.concatenate((inputs_new, np.ones((X_trn_raw.shape[0],1))), axis = 1) # add bias
 
X_tst_linear, y_tst = inputs[512:,:16], y_trn[512:]
X_trn_linear, y_trn = inputs[32:480,:16], y_trn[32:480]
X_trn_poly, X_tst_poly = inputs_new[32:480,:], inputs_new[512:,:]

pandas.DataFrame(inputs_new).to_csv("inputs.csv")

print(X_trn_poly.shape)
print(X_trn_linear.shape)


X_trn_linear = np.array([[float(elm) for elm in row] for row in X_trn_linear])
X_trn_linear = X_trn_linear.astype(np.float32)

w_linear = np.dot(np.linalg.inv(np.dot(X_trn_linear.T, X_trn_linear)), np.dot(X_trn_linear.T, y_trn))
w_poly = np.dot(np.linalg.inv(np.dot(X_trn_poly.T, X_trn_poly)), np.dot(X_trn_poly.T, y_trn))

#print(w_poly.shape)

y_pred_trn_linear = np.dot(w_linear, X_trn_linear.T)
y_pred_trn_poly = np.dot(w_poly, X_trn_poly.T)
y_pred_tst_linear = np.dot(w_linear, X_tst_linear.T)
y_pred_tst_poly = np.dot(w_poly, X_tst_poly.T)

l_trn_linear = np.sqrt(mean_squared_error(y_trn, y_pred_trn_linear))
l_tst_linear = np.sqrt(mean_squared_error(y_tst, y_pred_tst_linear))
l_trn_poly = np.sqrt(mean_squared_error(y_trn, y_pred_trn_poly))
l_tst_poly = np.sqrt(mean_squared_error(y_tst, y_pred_tst_poly))

pandas.DataFrame([y_trn, y_pred_trn_linear, y_pred_trn_poly, y_tst, y_pred_tst_linear, y_pred_tst_poly]).to_csv("new.csv")
#pandas.DataFrame([y_trn, y_pred_trn_linear, [], y_tst, y_pred_tst_linear]).to_csv("new.csv")

print('The average training loss using linear is: ', l_trn_linear)
print('The average testing loss using linear is: ', l_tst_linear)
print('The average training loss using poly is: ', l_trn_poly)
print('The average testing loss using poly is: ', l_tst_poly)
