import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
import math
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

pandas.DataFrame(X_trn_raw).to_csv("raw.csv")

for i in range(len(cover)):
    if cover[i] == -1:
        cover[i] = 0

X_trn_raw = (X_trn_raw * 1000).astype(int)
X_trn_raw = (X_trn_raw.astype(np.float32)/1000.0)
 
X_tst_linear, spread_tst, cover_tst, names_tst = X_trn_raw[544:,:], spread[544:], cover[544:], names[544:]
X_trn_linear, spread_trn, cover_trn, names_trn = X_trn_raw[94:480,:], spread[94:480], cover[94:480], names[94:480]

X_tst_linear = np.concatenate((X_tst_linear, np.array([spread_tst]).T), axis=1)
X_trn_linear = np.concatenate((X_trn_linear, np.array([spread_trn]).T), axis=1)

X_tst_linear_large = np.zeros(X_tst_linear.shape)
X_tst_linear_small = np.zeros(X_tst_linear.shape)
spread_tst_large = np.zeros(spread_tst.shape)
spread_tst_small = np.zeros(spread_tst.shape)
cover_tst_large = np.zeros(cover_tst.shape)
cover_tst_small = np.zeros(cover_tst.shape)
names_tst_large = np.ndarray(names_tst.shape, str)
names_tst_small = np.ndarray(names_tst.shape, str)
X_trn_linear_large = np.zeros(X_trn_linear.shape)
X_trn_linear_small = np.zeros(X_trn_linear.shape)
spread_trn_large = np.zeros(spread_trn.shape)
spread_trn_small = np.zeros(spread_trn.shape)
cover_trn_large = np.zeros(cover_trn.shape)
cover_trn_small = np.zeros(cover_trn.shape)
names_trn_large = np.ndarray(names_trn.shape, str)
names_trn_small = np.ndarray(names_trn.shape, str)

large_tst_count = 0
small_tst_count = 0
tst_count = 0
while tst_count < len(spread_tst):
    if abs(spread_tst[tst_count]) >= 5:
        X_tst_linear_large[large_tst_count,:] = X_tst_linear[tst_count,:]
        spread_tst_large[large_tst_count] = spread_tst[tst_count]
        cover_tst_large[large_tst_count] = cover_tst[tst_count]
        names_tst_large[large_tst_count] = names_tst[tst_count]
        large_tst_count+=1
    else:
        X_tst_linear_small[small_tst_count,:] = X_tst_linear[tst_count,:]
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
        spread_trn_large[large_trn_count] = spread_trn[trn_count]
        cover_trn_large[large_trn_count] = cover_trn[trn_count]
        names_trn_large[large_trn_count] = names_trn[trn_count]
        large_trn_count+=1
    else:
        X_trn_linear_small[small_trn_count,:] = X_trn_linear[trn_count,:]
        spread_trn_small[small_trn_count] = spread_trn[trn_count]
        cover_trn_small[small_trn_count] = cover_trn[trn_count]
        names_trn_small[small_trn_count] = names_trn[trn_count]
        small_trn_count+=1
    trn_count += 1


X_trn_linear_large = X_trn_linear_large[:large_trn_count,:]
spread_trn_large = spread_trn_large[:large_trn_count]
cover_trn_large = cover_trn_large[:large_trn_count]
names_trn_large = names_trn_large[:large_trn_count]
X_tst_linear_large = X_tst_linear_large[:large_tst_count,:]
spread_tst_large = spread_tst_large[:large_tst_count]
cover_tst_large = cover_tst_large[:large_tst_count]
names_tst_large = names_tst_large[:large_tst_count]
X_trn_linear_small = X_trn_linear_small[:small_trn_count,:]
spread_trn_small = spread_trn_small[:small_trn_count]
cover_trn_small = cover_trn_small[:small_trn_count]
names_trn_small = names_trn_small[:small_trn_count]
X_tst_linear_small = X_tst_linear_small[:small_tst_count,:]
spread_tst_small = spread_tst_small[:small_tst_count]
cover_tst_small = cover_tst_small[:small_tst_count]
names_tst_small = names_tst_small[:small_tst_count]

pandas.DataFrame(X_trn_linear).to_csv("trn_gen.csv")
print(np.any(np.isnan(X_trn_linear)))
print(np.all(np.isfinite(X_trn_linear)))
print(np.any(np.isnan(cover_trn)))
print(np.all(np.isfinite(cover_trn)))

class_large = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(40, 20, 40, 20), random_state=1, learning_rate='adaptive', max_iter=30000, learning_rate_init=.001)
class_small = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(40, 20, 40, 20), random_state=1, learning_rate='adaptive', max_iter=30000, learning_rate_init=.001)
class_gen = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(40, 20, 40, 20), random_state=1, learning_rate='adaptive', max_iter=30000, learning_rate_init=.001)

class_large.fit(X_trn_linear_large, cover_trn_large)
class_small.fit(X_trn_linear_small, cover_trn_small)
class_gen.fit(X_trn_linear, cover_trn)

tre1 = DecisionTreeClassifier(max_depth= 500, min_samples_split = 5, min_samples_leaf = 10)
tre2 = DecisionTreeClassifier(max_depth= 500, min_samples_split = 5, min_samples_leaf = 10)
tre3 = DecisionTreeClassifier(max_depth= 500, min_samples_split = 20, min_samples_leaf = 20)
tre1.fit(X_trn_linear_large, cover_trn_large)
tre2.fit(X_trn_linear_small, cover_trn_small)
tre3.fit(X_trn_linear, cover_trn)

clf1 = AdaBoostClassifier(base_estimator=tre1, n_estimators=500).fit(X_trn_linear_large, cover_trn_large)
clf2 = AdaBoostClassifier(base_estimator=tre2, n_estimators=500).fit(X_trn_linear_small, cover_trn_small)
clf3 = AdaBoostClassifier(base_estimator=tre3, n_estimators=1000).fit(X_trn_linear, cover_trn)

cover_pred_large_tst = clf1.predict(X_tst_linear_large)
cover_pred_small_tst = clf2.predict(X_tst_linear_small)
cover_pred_gen_tst = clf3.predict(X_tst_linear)

cover_pred_large_trn = clf1.predict(X_trn_linear_large)
cover_pred_small_trn = clf2.predict(X_trn_linear_small)
cover_pred_gen_trn = clf3.predict(X_trn_linear)


large_cover_trn_correct = 0
large_cover_trn_incorrect = 0
small_cover_trn_correct = 0
small_cover_trn_incorrect = 0
gen_cover_trn_correct = 0
gen_cover_trn_incorrect = 0
large_cover_tst_correct = 0
large_cover_tst_incorrect = 0
small_cover_tst_correct = 0
small_cover_tst_incorrect = 0
gen_cover_tst_correct = 0
gen_cover_tst_incorrect = 0

for i in range(len(cover_pred_large_trn)):
    if cover_trn_large[i] == cover_pred_large_trn[i]:
        large_cover_trn_correct += 1

for i in range(len(cover_pred_small_trn)):
    if cover_trn_small[i] == cover_pred_small_trn[i]:
        small_cover_trn_correct += 1

for i in range(len(cover_pred_gen_trn)):
    if cover_trn[i] == cover_pred_gen_trn[i]:
        gen_cover_trn_correct += 1

for i in range(len(cover_pred_large_tst)):
    if cover_tst_large[i] == cover_pred_large_tst[i]:
        large_cover_tst_correct += 1

for i in range(len(cover_pred_small_tst)):
    if cover_tst_small[i] == cover_pred_small_tst[i]:
        small_cover_tst_correct += 1

for i in range(len(cover_pred_small_tst)):
    if cover_tst[i] == cover_pred_gen_tst[i]:
        gen_cover_tst_correct += 1

large_cover_trn_incorrect = len(cover_pred_large_trn) - large_cover_trn_correct
small_cover_trn_incorrect = len(cover_pred_small_trn) - small_cover_trn_correct
gen_cover_trn_incorrect = len(cover_pred_gen_trn) - gen_cover_trn_correct
large_cover_tst_incorrect = len(cover_pred_large_tst) - large_cover_tst_correct
small_cover_tst_incorrect = len(cover_pred_small_tst) - small_cover_tst_correct
gen_cover_tst_incorrect = len(cover_pred_small_tst) - gen_cover_tst_correct

pandas.DataFrame([cover_tst,cover_pred_gen_tst, cover_tst_large, cover_pred_large_tst, cover_tst_small, cover_pred_small_tst]).to_csv("finals.csv")

print('Percent of correct large trn ', float(large_cover_trn_correct)/float(len(cover_pred_large_trn)))
print('Percent of correct small trn ', float(small_cover_trn_correct)/float(len(cover_pred_small_trn)))
print('Percent of correct gen trn ', float(gen_cover_trn_correct)/float(len(cover_pred_gen_trn)))
print('Percent of correct large tst ', float(large_cover_tst_correct)/float(len(cover_pred_large_tst)))
print('Percent of correct small tst ', float(small_cover_tst_correct)/float(len(cover_pred_small_tst)))
print('Percent of correct gen tst ', float(gen_cover_tst_correct)/float(len(cover_pred_gen_tst)))


