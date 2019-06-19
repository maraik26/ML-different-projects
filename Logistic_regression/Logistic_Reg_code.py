
# coding: utf-8

#!/usr/bin/env python
#get_ipython().magic('matplotlib inline')
# Run logistic regression training for different learning rates with stochastic gradient descent.
import pandas as pd
import numpy as np
import scipy.special as sps
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
#import assignment2 as a2

#execfile("logistic_regression_mod.py")
data=pd.read_csv("preprocessed_datasets.csv")
# Randomly permute the training data.
#data = np.random.permutation(data)

draft_year=data["DraftYear"]
GP_greater_than_0_ori=data["GP_greater_than_0"]
data.drop(["id", "PlayerName", "sum_7yr_TOI", "Overall","Country","sum_7yr_GP"], axis=1, inplace=True)

mask_no = GP_greater_than_0_ori == 'no'
GP_greater_than_0_ori[mask_no] = 0
mask_yes = GP_greater_than_0_ori == 'yes'
GP_greater_than_0_ori[mask_yes] = 1
GP_greater_than_0_ori = GP_greater_than_0_ori.astype(np.float64)

def standardize(df):
    res = df.copy()
    for i in res.columns:
        mean=np.mean(res[i])
        sd=np.std(res[i])
        if(sd == 0):
             res.drop(i, axis=1, inplace=True)
        else:
             res[i]=(res[i]-mean)/sd
    return res

def insert_1(df):
    res = df.copy()
    res['X'] = 1
    return res

train_set=data[data['DraftYear'].isin([2004,2005,2006])]
test_set=data[data['DraftYear'].isin([2007])]

GP_greater_than_0_train=train_set["GP_greater_than_0"].astype(np.float64)
GP_greater_than_0_test=test_set["GP_greater_than_0"].astype(np.float64)

train_set.drop(['DraftYear',"GP_greater_than_0"], axis=1, inplace=True)
test_set.drop(['DraftYear',"GP_greater_than_0"], axis=1, inplace=True)

train_set = pd.get_dummies(train_set, columns= ['country_group','Position'])
test_set = pd.get_dummies(test_set, columns= ['country_group','Position'])
#data.drop(['DraftYear',"GP_greater_than_0"], axis=1, inplace=True)

#train_set_interaction = interaction(train_set)
train_set_std = standardize(train_set)
test_set_std = standardize(test_set)

train_set_final = insert_1(train_set_std)
test_set_final = insert_1(test_set_std)

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 300
tol = 0.00001

# Target values, 0 for class 1, 1 for class 2.
#GP_greater_than_0_ori=data["GP_greater_than_0"]
t = len(data)

n_train=data['GP_greater_than_0'].size

# Error values over all iterations.
all_errors = dict()

m = len(GP_greater_than_0_train)

# Step size for gradient descent.
etas = [1.9, 1.0, 0.5, 0.3, 0.1, 0.05, 0.01]
#etas = [0.1, 0.05, 0.01]

all_errors = {}
accs = {}

for eta in etas:
        # Initialize w.
    w = np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((23,1))
    e_all = []
    for iter in range (0, max_iter):
        # Compute output using current w on sample x_n.
        #p =np.dot(train_set_final,w)
        y_train = sps.expit(np.dot(train_set_final,w)).transpose()

        # Gradient of the error, using Assignment result
        grad_e = np.dot((y_train - np.array(GP_greater_than_0_train).reshape(1, m)), train_set_final)
      
        # Update w, *subtracting* a step in the error derivative since we're minimizing
        # w = fill this in
        w = w - (1.0 * eta / m) * grad_e.reshape((23, 1))
        
        # Compute error over all examples, add this error to the end of error vector.
        # Compute output using current w on all data X.
        y_train = sps.expit(np.dot(train_set_final,w)).transpose()
        
        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(GP_greater_than_0_train.reshape(1,m), np.log(y_train)) 
                     + np.multiply((1-GP_greater_than_0_train.reshape(1,m)),np.log(1-y_train)))
        e_all.append(e)

        # Print some information.
        print ('eta={0}, epoch {1:d}, negative log-likelihood {2:.4f}, w={3}'.format(eta, iter, e, w.T))
        #Stop iterating if error doesn't change more than tol.
        if iter>0:
            if np.absolute(e-e_all[iter-1]) < tol:
                break
            
    all_errors[eta] = e_all
    y_test = sps.expit(np.dot(test_set_final,w)).transpose()
    preds = np.where( y_test > 0.5, 1, 0)
    accuracy = 1 - np.mean((preds - GP_greater_than_0_test.reshape(1,191))**2)
    accs[eta] = accuracy
    print(accs)

plt.figure(figsize=(16,6))
for eta in list(all_errors.keys()):
    plt.plot(all_errors[eta], label='eta={}'.format(eta))
plt.legend()
plt.ylabel('Cross-Entropy(negative log-likelihood)')
plt.title('Cross-Entropy after each epoch for each learning rate')
plt.xlabel('Epochs')
plt.savefig('Picture.png')

