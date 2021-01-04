import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

ree = 0
getter1 = pandas.read_csv('test-nfl-spread.csv')
getter2 = getter1.values
train = np.zeros([1, getter2.shape[1]])
for i in range(len(getter2)):
    try:
        if not getter2[i, 3] == '-' and not math.isnan(float(getter2[i, 3])):
            train = np.vstack((train, getter2[i, :]))
    except:
        pass

X_trn_raw, y_trn = train[1:,3:21], train[1:,23]
spread = train[1:,30]
cover = train[1:,31]
names = train[1:,1]
y_trn = np.array([float(item) for item in y_trn])
y_trn = y_trn.astype(np.float32)
spread = np.array([float(item) for item in spread])
spread = spread.astype(np.float32)
cover = np.array([int(item) for item in cover])
cover = cover.astype(np.int)
X_trn_raw = np.array([[float(elm) for elm in row] for row in X_trn_raw])
X_trn_raw = X_trn_raw.astype(np.float32)
for i in range(len(X_trn_raw)):
    X_trn_raw[i, 1] = X_trn_raw[i, 0] - X_trn_raw[i, 1]
    X_trn_raw[i, 10] = X_trn_raw[i, 9] - X_trn_raw[i, 10]
    # X_trn_raw[i, 0:7] = X_trn_raw[i, 0:7] * (2 + X_trn_raw[i, 7])
    # X_trn_raw[i, 8:15] = X_trn_raw[i, 8:15] * (2 + X_trn_raw[i, 15])

pandas.DataFrame(X_trn_raw).to_csv("raw.csv")

inputs = X_trn_raw - np.mean(X_trn_raw,axis=0) #shift
inputs = inputs/(np.max(inputs,axis=0)) #normalize
inputs = np.concatenate((inputs, np.ones((X_trn_raw.shape[0],1))), axis = 1) #add bias

# inputs_new = np.concatenate((X_trn_raw[:,0:9],np.square(X_trn_raw[:,0:9])), axis=1) #add square term
# inputs_new = inputs_new - np.mean(inputs_new,axis=0) #shift
# inputs_new = inputs_new/(np.max(inputs_new,axis=0)) #normalize
# inputs_new_bruteforce = np.zeros(inputs_new.shape)
#
while ree < len(X_trn_raw):
#     inputs_new_bruteforce[ree, :] = inputs_new[ree+1, :]
#     inputs_new_bruteforce[ree+1, :] = inputs_new[ree, :]
    inputs[ree, 9:18] = inputs[ree+1, 0:9]
    inputs[ree+1, 9:18] = inputs[ree, 0:9]
    ree += 2

print(inputs.shape)
#
# inputs_new = np.concatenate((inputs_new, inputs_new_bruteforce), axis=1)
# inputs_new = np.concatenate((inputs_new, np.ones((X_trn_raw.shape[0],1))), axis = 1) # add bias

#pandas.DataFrame(inputs_new).to_csv("inputs.csv")
 
X_tst_linear, y_tst, spread_tst, cover_tst, names_tst = inputs[544:,:18], y_trn[544:], spread[544:], cover[544:], names[544:]
X_trn_linear, y_trn, spread_trn, cover_trn, names_trn = inputs[94:480,:18], y_trn[94:480], spread[94:480], cover[94:480], names[94:480]
#X_trn_poly, X_tst_poly = inputs_new[94:480,:], inputs_new[544:,:]

X_tst_linear_large = np.zeros(X_tst_linear.shape)
X_tst_linear_small = np.zeros(X_tst_linear.shape)
y_tst_large = np.zeros(y_tst.shape)
y_tst_small = np.zeros(y_tst.shape)
spread_tst_large = np.zeros(spread_tst.shape)
spread_tst_small = np.zeros(spread_tst.shape)
cover_tst_large = np.zeros(cover_tst.shape)
cover_tst_small = np.zeros(cover_tst.shape)
names_tst_large = np.ndarray(names_tst.shape, str)
names_tst_small = np.ndarray(names_tst.shape, str)
X_tst_poly_large = np.zeros(X_tst_linear.shape)
X_tst_poly_small = np.zeros(X_tst_linear.shape)
X_trn_linear_large = np.zeros(X_trn_linear.shape)
X_trn_linear_small = np.zeros(X_trn_linear.shape)
y_trn_large = np.zeros(y_trn.shape)
y_trn_small = np.zeros(y_trn.shape)
spread_trn_large = np.zeros(spread_trn.shape)
spread_trn_small = np.zeros(spread_trn.shape)
cover_trn_large = np.zeros(cover_trn.shape)
cover_trn_small = np.zeros(cover_trn.shape)
names_trn_large = np.ndarray(names_trn.shape, str)
names_trn_small = np.ndarray(names_trn.shape, str)
X_trn_poly_large = np.zeros(X_trn_linear.shape)
X_trn_poly_small = np.zeros(X_trn_linear.shape)

large_tst_count = 0
small_tst_count = 0
tst_count = 0
while tst_count < len(spread_tst):
    if abs(spread_tst[tst_count]) >= 5:
        X_tst_linear_large[large_tst_count,:] = X_tst_linear[tst_count,:]
        X_tst_poly_large[large_tst_count,:] = X_tst_linear[tst_count,:]
        y_tst_large[large_tst_count] = y_tst[tst_count]
        spread_tst_large[large_tst_count] = spread_tst[tst_count]
        cover_tst_large[large_tst_count] = cover_tst[tst_count]
        names_tst_large[large_tst_count] = names_tst[tst_count]
        large_tst_count+=1
    else:
        X_tst_linear_small[small_tst_count,:] = X_tst_linear[tst_count,:]
        X_tst_poly_small[small_tst_count,:] = X_tst_linear[tst_count,:]
        y_tst_small[small_tst_count] = y_tst[tst_count]
        spread_tst_small[small_tst_count] = spread_tst[tst_count]
        cover_tst_small[small_tst_count] = cover_tst[tst_count]
        names_tst_small[small_tst_count] = names_tst[tst_count]
        small_tst_count+=1
    tst_count += 1

large_trn_count = 0
small_trn_count = 0
trn_count = 0
while trn_count < len(spread_trn):
    if abs(spread_trn[trn_count]) >= 5:
        X_trn_linear_large[large_trn_count,:] = X_trn_linear[trn_count,:]
        X_trn_poly_large[large_trn_count,:] = X_trn_linear[trn_count,:]
        y_trn_large[large_trn_count] = y_trn[trn_count]
        spread_trn_large[large_trn_count] = spread_trn[trn_count]
        cover_trn_large[large_trn_count] = cover_trn[trn_count]
        names_trn_large[large_trn_count] = names_trn[trn_count]
        large_trn_count+=1
    else:
        X_trn_linear_small[small_trn_count,:] = X_trn_linear[trn_count,:]
        X_trn_poly_small[small_trn_count,:] = X_trn_linear[trn_count,:]
        y_trn_small[small_trn_count] = y_trn[trn_count]
        spread_trn_small[small_trn_count] = spread_trn[trn_count]
        cover_trn_small[small_trn_count] = cover_trn[trn_count]
        names_trn_small[small_trn_count] = names_trn[trn_count]
        small_trn_count+=1
    trn_count += 1

X_trn_linear_large = X_trn_linear_large[:large_trn_count,:]
X_trn_poly_large = X_trn_poly_large[:large_trn_count,:]
y_trn_large = y_trn_large[:large_trn_count]
spread_trn_large = spread_trn_large[:large_trn_count]
cover_trn_large = cover_trn_large[:large_trn_count]
names_trn_large = names_trn_large[:large_trn_count]
X_tst_linear_large = X_tst_linear_large[:large_tst_count,:]
X_tst_poly_large = X_tst_poly_large[:large_tst_count,:]
y_tst_large = y_tst_large[:large_tst_count]
spread_tst_large = spread_tst_large[:large_tst_count]
cover_tst_large = cover_tst_large[:large_tst_count]
names_tst_large = names_tst_large[:large_tst_count]
X_trn_linear_small = X_trn_linear_small[:small_trn_count,:]
X_trn_poly_small = X_trn_poly_small[:small_trn_count,:]
y_trn_small = y_trn_small[:small_trn_count]
spread_trn_small = spread_trn_small[:small_trn_count]
cover_trn_small = cover_trn_small[:small_trn_count]
names_trn_small = names_trn_small[:small_trn_count]
X_tst_linear_small = X_tst_linear_small[:small_tst_count,:]
X_tst_poly_small = X_tst_poly_small[:small_tst_count,:]
y_tst_small = y_tst_small[:small_tst_count]
spread_tst_small = spread_tst_small[:small_tst_count]
cover_tst_small = cover_tst_small[:small_tst_count]
names_tst_small = names_tst_small[:small_tst_count]

inputs_new_trn_large = np.concatenate((X_trn_linear_large[:,0:9],np.square(X_trn_linear_large[:,0:9])), axis=1) #add square term
inputs_new_trn_large = inputs_new_trn_large - np.mean(inputs_new_trn_large,axis=0) #shift
inputs_new_trn_large = inputs_new_trn_large/(np.max(inputs_new_trn_large,axis=0)) #normalize
inputs_new_bruteforce_trn_large = np.zeros(inputs_new_trn_large.shape)


fix_trn_large = 0
while fix_trn_large < len(X_trn_linear_large):
    inputs_new_bruteforce_trn_large[fix_trn_large, :] = inputs_new_trn_large[fix_trn_large+1, :]
    inputs_new_bruteforce_trn_large[fix_trn_large+1, :] = inputs_new_trn_large[fix_trn_large, :]
    fix_trn_large += 2

X_trn_poly_large = np.concatenate((inputs_new_trn_large, inputs_new_bruteforce_trn_large), axis=1)
X_trn_poly_large = np.concatenate((X_trn_poly_large, np.ones((X_trn_linear_large.shape[0],1))), axis = 1) # add bias

inputs_new_trn_small = np.concatenate((X_trn_linear_small[:,0:9],np.square(X_trn_linear_small[:,0:9])), axis=1) #add square term
inputs_new_trn_small = inputs_new_trn_small - np.mean(inputs_new_trn_small,axis=0) #shift
inputs_new_trn_small = inputs_new_trn_small/(np.max(inputs_new_trn_small,axis=0)) #normalize
inputs_new_bruteforce_trn_small = np.zeros(inputs_new_trn_small.shape)


fix_trn_small = 0
while fix_trn_small < len(X_trn_linear_small):
    inputs_new_bruteforce_trn_small[fix_trn_small, :] = inputs_new_trn_small[fix_trn_small+1, :]
    inputs_new_bruteforce_trn_small[fix_trn_small+1, :] = inputs_new_trn_small[fix_trn_small, :]
    fix_trn_small += 2

X_trn_poly_small = np.concatenate((inputs_new_trn_small, inputs_new_bruteforce_trn_small), axis=1)
X_trn_poly_small = np.concatenate((X_trn_poly_small, np.ones((X_trn_linear_small.shape[0],1))), axis = 1) # add bias

inputs_new_tst_large = np.concatenate((X_tst_linear_large[:,0:9],np.square(X_tst_linear_large[:,0:9])), axis=1) #add square term
inputs_new_tst_large = inputs_new_tst_large - np.mean(inputs_new_tst_large,axis=0) #shift
inputs_new_tst_large = inputs_new_tst_large/(np.max(inputs_new_tst_large,axis=0)) #normalize
inputs_new_bruteforce_tst_large = np.zeros(inputs_new_tst_large.shape)


fix_tst_large = 0
while fix_tst_large < len(X_tst_linear_large):
    inputs_new_bruteforce_tst_large[fix_tst_large, :] = inputs_new_tst_large[fix_tst_large+1, :]
    inputs_new_bruteforce_tst_large[fix_tst_large+1, :] = inputs_new_tst_large[fix_tst_large, :]
    fix_tst_large += 2

X_tst_poly_large = np.concatenate((inputs_new_tst_large, inputs_new_bruteforce_tst_large), axis=1)
X_tst_poly_large = np.concatenate((X_tst_poly_large, np.ones((X_tst_linear_large.shape[0],1))), axis = 1) # add bias

inputs_new_tst_small = np.concatenate((X_tst_linear_small[:,0:9],np.square(X_tst_linear_small[:,0:9])), axis=1) #add square term
inputs_new_tst_small = inputs_new_tst_small - np.mean(inputs_new_tst_small,axis=0) #shift
inputs_new_tst_small = inputs_new_tst_small/(np.max(inputs_new_tst_small,axis=0)) #normalize
inputs_new_bruteforce_tst_small = np.zeros(inputs_new_tst_small.shape)


fix_tst_small = 0
while fix_tst_small < len(X_tst_linear_small):
    inputs_new_bruteforce_tst_small[fix_tst_small, :] = inputs_new_tst_small[fix_tst_small+1, :]
    inputs_new_bruteforce_tst_small[fix_tst_small+1, :] = inputs_new_tst_small[fix_tst_small, :]
    fix_tst_small += 2

X_tst_poly_small = np.concatenate((inputs_new_tst_small, inputs_new_bruteforce_tst_small), axis=1)
X_tst_poly_small = np.concatenate((X_tst_poly_small, np.ones((X_tst_linear_small.shape[0],1))), axis = 1) # add bias

# for i in range(len(y_trn_large)):
#     X_trn_raw[i, 0:7] = X_trn_raw[i, 0:7] * (2 + X_trn_raw[i, 7])
#     X_trn_raw[i, 8:15] = X_trn_raw[i, 8:15] * (2 + X_trn_raw[i, 15])

# w_optimal_linear = []
# w_optimal_poly = []
# lamb = []
# points = 100
# for i in range(points):
#     lamb.append(0.001*(1.2**i))
# err_trn_linear = []
# err_tst_linear = []
# err_trn_poly = []
# err_tst_poly = []
#
# for i in range(points):
#     w_optimal_linear.append(np.dot(np.linalg.inv(np.dot(X_trn_linear_large.T, X_trn_linear_large)+lamb[i]*np.identity(X_trn_linear_large.shape[1])), np.dot(X_trn_linear_large.T, y_trn_large)))
#     w_optimal_poly.append(np.dot(np.linalg.inv(np.dot(X_trn_poly_large.T, X_trn_poly_large)+lamb[i]*np.identity(X_trn_poly_large.shape[1])), np.dot(X_trn_poly_large.T, y_trn_large)))
#     err_trn_linear.append(np.sqrt(mean_squared_error(y_trn_large, np.dot(w_optimal_linear[i], X_trn_linear_large.T))))
#     err_tst_linear.append(np.sqrt(mean_squared_error(y_tst_large, np.dot(w_optimal_linear[i], X_tst_linear_large.T))))
#     err_trn_poly.append(np.sqrt(mean_squared_error(y_trn_large, np.dot(w_optimal_poly[i], X_trn_poly_large.T))))
#     err_tst_poly.append(np.sqrt(mean_squared_error(y_tst_large, np.dot(w_optimal_poly[i], X_tst_poly_large.T))))
#
# temp1 = min(err_tst_linear)
# temp2 = min(err_tst_poly)
# min_linear = 0
# min_poly = 0
# for i in range(len(err_tst_poly)):
#     if err_tst_linear[i] == temp1:
#         min_linear = i
#     if err_tst_poly[i] == temp2:
#         min_poly = i
#
# w_linear_large = w_optimal_linear[min_linear]
# w_poly_large = w_optimal_poly[min_poly]

print(X_trn_poly_large.shape)
print(y_trn_large.shape)
print(X_trn_poly_small.shape)
print(y_trn_small.shape)

w_linear_large = np.dot(np.linalg.inv(np.dot(X_trn_linear_large.T, X_trn_linear_large)), np.dot(X_trn_linear_large.T, y_trn_large))
w_poly_large = np.dot(np.linalg.inv(np.dot(X_trn_poly_large.T, X_trn_poly_large)), np.dot(X_trn_poly_large.T, y_trn_large))
w_linear_small = np.dot(np.linalg.inv(np.dot(X_trn_linear_small.T, X_trn_linear_small)), np.dot(X_trn_linear_small.T, y_trn_small))
w_poly_small = np.dot(np.linalg.inv(np.dot(X_trn_poly_small.T, X_trn_poly_small)), np.dot(X_trn_poly_small.T, y_trn_small))

pandas.DataFrame(w_poly_large).to_csv("junk.csv")


y_pred_trn_linear_large = np.dot(w_linear_large, X_trn_linear_large.T)
y_pred_trn_poly_large = np.dot(w_poly_large, X_trn_poly_large.T)
y_pred_tst_linear_large = np.dot(w_linear_large, X_tst_linear_large.T)
y_pred_tst_poly_large = np.dot(w_poly_large, X_tst_poly_large.T)
y_pred_trn_linear_small = np.dot(w_linear_small, X_trn_linear_small.T)
y_pred_trn_poly_small = np.dot(w_poly_small, X_trn_poly_small.T)
y_pred_tst_linear_small = np.dot(w_linear_small, X_tst_linear_small.T)
y_pred_tst_poly_small = np.dot(w_poly_small, X_tst_poly_small.T)

l_trn_linear_large = np.sqrt(mean_squared_error(y_trn_large, y_pred_trn_linear_large))
l_tst_linear_large = np.sqrt(mean_squared_error(y_tst_large[:42], y_pred_tst_linear_large[:42]))
l_trn_poly_large = np.sqrt(mean_squared_error(y_trn_large, y_pred_trn_poly_large))
l_tst_poly_large = np.sqrt(mean_squared_error(y_tst_large[:42], y_pred_tst_poly_large[:42]))
l_trn_linear_small = np.sqrt(mean_squared_error(y_trn_small, y_pred_trn_linear_small))
l_tst_linear_small = np.sqrt(mean_squared_error(y_tst_small[:46], y_pred_tst_linear_small[:46]))
l_trn_poly_small = np.sqrt(mean_squared_error(y_trn_small, y_pred_trn_poly_small))
l_tst_poly_small = np.sqrt(mean_squared_error(y_tst_small[:46], y_pred_tst_poly_small[:46]))

cover_pred_trn_linear_large = np.sign(spread_trn_large + y_pred_trn_linear_large)
cover_pred_trn_poly_large = np.sign(spread_trn_large + y_pred_trn_poly_large)
cover_pred_tst_linear_large = np.sign(spread_tst_large + y_pred_tst_linear_large)
cover_pred_tst_poly_large = np.sign(spread_tst_large + y_pred_tst_poly_large)
cover_pred_trn_linear_small = np.sign(spread_trn_small + y_pred_trn_linear_small)
cover_pred_trn_poly_small = np.sign(spread_trn_small + y_pred_trn_poly_small)
cover_pred_tst_linear_small = np.sign(spread_tst_small + y_pred_tst_linear_small)
cover_pred_tst_poly_small = np.sign(spread_tst_small + y_pred_tst_poly_small)


large_cover_trn_linear_correct = 0
large_cover_trn_linear_incorrect = 0
large_cover_trn_poly_correct = 0
large_cover_trn_poly_incorrect = 0
large_cover_tst_linear_correct = 0
large_cover_tst_linear_incorrect = 0
large_cover_tst_poly_correct = 0
large_cover_tst_poly_incorrect = 0
small_cover_trn_linear_correct = 0
small_cover_trn_linear_incorrect = 0
small_cover_trn_poly_correct = 0
small_cover_trn_poly_incorrect = 0
small_cover_tst_linear_correct = 0
small_cover_tst_linear_incorrect = 0
small_cover_tst_poly_correct = 0
small_cover_tst_poly_incorrect = 0

for i in range(len(cover_pred_trn_linear_large)):
    if cover_trn_large[i] == 0:
        large_cover_trn_linear_correct += 0
    elif cover_pred_trn_linear_large[i] == cover_trn_large[i]:
        large_cover_trn_linear_correct += 1
    else:
        large_cover_trn_linear_incorrect += 1

for i in range(len(cover_pred_trn_poly_large)):
    if cover_trn_large[i] == 0:
        large_cover_trn_poly_correct += 0
    elif cover_pred_trn_poly_large[i] == cover_trn_large[i]:
        large_cover_trn_poly_correct += 1
    else:
        large_cover_trn_poly_incorrect += 1

for i in range(len(cover_pred_tst_linear_large)):
    if cover_tst_large[i] == 0:
        large_cover_tst_linear_correct += 0
    elif cover_pred_tst_linear_large[i] == cover_tst_large[i]:
        large_cover_tst_linear_correct += 1
    else:
        large_cover_tst_linear_incorrect += 1

for i in range(len(cover_pred_tst_poly_large)):
    if cover_tst_large[i] == 0:
        large_cover_tst_poly_correct += 0
    elif cover_pred_tst_poly_large[i] == cover_tst_large[i]:
        large_cover_tst_poly_correct += 1
    else:
        large_cover_tst_poly_incorrect += 1

for i in range(len(cover_pred_trn_linear_small)):
    if cover_trn_small[i] == 0:
        small_cover_trn_linear_correct += 0
    elif cover_pred_trn_linear_small[i] == cover_trn_small[i]:
        small_cover_trn_linear_correct += 1
    else:
        small_cover_trn_linear_incorrect += 1

for i in range(len(cover_pred_trn_poly_small)):
    if cover_trn_small[i] == 0:
        small_cover_trn_poly_correct += 0
    elif cover_pred_trn_poly_small[i] == cover_trn_small[i]:
        small_cover_trn_poly_correct += 1
    else:
        small_cover_trn_poly_incorrect += 1

for i in range(len(cover_pred_tst_linear_small)):
    if cover_tst_small[i] == 0:
        small_cover_tst_linear_correct += 0
    elif cover_pred_tst_linear_small[i] == cover_tst_small[i]:
        small_cover_tst_linear_correct += 1
    else:
        small_cover_tst_linear_incorrect += 1

for i in range(len(cover_pred_tst_poly_small)):
    if cover_tst_small[i] == 0:
        small_cover_tst_poly_correct += 0
    elif cover_pred_tst_poly_small[i] == cover_tst_small[i]:
        small_cover_tst_poly_correct += 1
    else:
        small_cover_tst_poly_incorrect += 1

large_cover_trn_linear_percent = float(large_cover_trn_linear_correct)/(float(large_cover_trn_linear_correct)+float(large_cover_trn_linear_incorrect))
large_cover_trn_poly_percent = float(large_cover_trn_poly_correct)/(float(large_cover_trn_poly_correct)+float(large_cover_trn_poly_incorrect))
large_cover_tst_linear_percent = float(large_cover_tst_linear_correct)/(float(large_cover_tst_linear_correct)+float(large_cover_tst_linear_incorrect))
large_cover_tst_poly_percent = float(large_cover_tst_poly_correct)/(float(large_cover_tst_poly_correct)+float(large_cover_tst_poly_incorrect))
small_cover_trn_linear_percent = float(small_cover_trn_linear_correct)/(float(small_cover_trn_linear_correct)+float(small_cover_trn_linear_incorrect))
small_cover_trn_poly_percent = float(small_cover_trn_poly_correct)/(float(small_cover_trn_poly_correct)+float(small_cover_trn_poly_incorrect))
small_cover_tst_linear_percent = float(small_cover_tst_linear_correct)/(float(small_cover_tst_linear_correct)+float(small_cover_tst_linear_incorrect))
small_cover_tst_poly_percent = float(small_cover_tst_poly_correct)/(float(small_cover_tst_poly_correct)+float(small_cover_tst_poly_incorrect))

cover_trn_linear_correct = large_cover_trn_linear_correct + small_cover_trn_linear_correct
cover_trn_poly_correct = large_cover_trn_poly_correct + small_cover_trn_poly_correct
cover_tst_linear_correct = large_cover_tst_linear_correct + small_cover_tst_linear_correct
cover_tst_poly_correct = large_cover_tst_poly_correct + small_cover_tst_poly_correct
cover_trn_linear_incorrect = large_cover_trn_linear_incorrect + small_cover_trn_linear_incorrect
cover_trn_poly_incorrect = large_cover_trn_poly_incorrect + small_cover_trn_poly_incorrect
cover_tst_linear_incorrect = large_cover_tst_linear_incorrect + small_cover_tst_linear_incorrect
cover_tst_poly_incorrect = large_cover_tst_poly_incorrect + small_cover_tst_poly_incorrect

cover_trn_linear_percent = float(cover_trn_linear_correct)/(float(cover_trn_linear_correct)+float(cover_trn_linear_incorrect))
cover_trn_poly_percent = float(cover_trn_poly_correct)/(float(cover_trn_poly_correct)+float(cover_trn_poly_incorrect))
cover_tst_linear_percent = float(cover_tst_linear_correct)/(float(cover_tst_linear_correct)+float(cover_tst_linear_incorrect))
cover_tst_poly_percent = float(cover_tst_poly_correct)/(float(cover_tst_poly_correct)+float(cover_tst_poly_incorrect))

pandas.DataFrame([names_tst_large, spread_tst_large, y_tst_large, y_pred_tst_poly_large, cover_tst_large, \
                 cover_pred_tst_poly_large, names_tst_small, spread_tst_small, y_tst_small, y_pred_tst_linear_small, \
                  cover_tst_small, cover_pred_tst_linear_small]).to_csv("new.csv")



print('The average training loss using linear for large spread is: ', l_trn_linear_large)
print('The average testing loss using linear for large spread is: ', l_tst_linear_large)
print('The average training loss using poly for large spread is: ', l_trn_poly_large)
print('The average testing loss using poly for large spread is: ', l_tst_poly_large)
print('The average training loss using linear for small spread is: ', l_trn_linear_small)
print('The average testing loss using linear for small spread is: ', l_tst_linear_small)
print('The average training loss using poly for small spread is: ', l_trn_poly_small)
print('The average testing loss using poly for small spread is: ', l_tst_poly_small)

# print('The percent training correct using linear is: ', cover_trn_linear_percent)
# print('The percent training correct using poly is: ', cover_trn_poly_percent)
# print('The percent testing correct using linear is: ', cover_tst_linear_percent)
# print('The percent testing correct using poly is: ', cover_tst_poly_percent)


print('The percent training correct using linear for large spread is: ', large_cover_trn_linear_percent)
print('The percent training correct using poly for large spread is: ', large_cover_trn_poly_percent)
print('The percent testing correct using linear for large spread is: ', large_cover_tst_linear_percent)
print('The percent testing correct using poly for large spread is: ', large_cover_tst_poly_percent)

print('The percent training correct using linear for small spread is: ', small_cover_trn_linear_percent)
print('The percent training correct using poly for small spread is: ', small_cover_trn_poly_percent)
print('The percent testing correct using linear for small spread is: ', small_cover_tst_linear_percent)
print('The percent testing correct using poly for small spread is: ', small_cover_tst_poly_percent)


print(large_cover_trn_linear_correct, large_cover_trn_linear_incorrect)
print(large_cover_trn_poly_correct, large_cover_trn_poly_incorrect)
print(large_cover_tst_linear_correct, large_cover_tst_linear_incorrect)
print(large_cover_tst_poly_correct, large_cover_tst_poly_incorrect)
print(small_cover_trn_linear_correct, small_cover_trn_linear_incorrect)
print(small_cover_trn_poly_correct, small_cover_trn_poly_incorrect)
print(small_cover_tst_linear_correct, small_cover_tst_linear_incorrect)
print(small_cover_tst_poly_correct, small_cover_tst_poly_incorrect)

