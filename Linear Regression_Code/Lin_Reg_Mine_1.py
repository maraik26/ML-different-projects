import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

data=pd.read_csv("C:/Users/Irina/Documents/SFU/Machine_Learning/Assignement2/preprocessed_datasets.csv")
draft_year=data["DraftYear"]
sum_7yr_gp_ori=data["sum_7yr_GP"]
data.drop(["id", "PlayerName", "sum_7yr_TOI", "Overall", "GP_greater_than_0","Country"], axis=1, inplace=True)

def interaction(dff1):
    data2=pd.DataFrame()
    for i in range(len(dff1.columns)):
        for j in range(i+1,len(dff1.columns)):
            col_name=str(str(dff1.columns[i])+":"+str(dff1.columns[j]))
            data2[col_name]=dff1.iloc[:,i]*dff1.iloc[:,j]
    dff1=pd.concat([dff1,data2], axis=1)
    return dff1

def standardize(dff2):
    for i in dff2.columns:
        mean=np.mean(dff2[i])
        sd=np.std(dff2[i])
    
        if(sd == 0):
            dff2.drop(i,axis=1, inplace=True)
        else:
            dff2[i]=(dff2[i]-mean)/sd
    return dff2
        
def insert_1(dff3):
    dff3['X'] = 1
    cols = dff3.columns.tolist()
    dff3 = dff3[[cols[-1]] + cols[:-1]]
    return dff3

train_set=data[data['DraftYear'].isin([2004,2005,2006])]
test_set=data[data['DraftYear'].isin([2007])]
sum_7yr_gp_train=train_set["sum_7yr_GP"]
sum_7yr_gp_test=test_set["sum_7yr_GP"]
train_set.drop(['DraftYear',"sum_7yr_GP"], axis=1, inplace=True)
test_set.drop(['DraftYear',"sum_7yr_GP"], axis=1, inplace=True)
train_set = pd.get_dummies(train_set, columns= ['country_group','Position'])
test_set = pd.get_dummies(test_set, columns= ['country_group','Position'])
data.drop(['DraftYear',"sum_7yr_GP"], axis=1, inplace=True)

train_set_interaction = interaction(train_set)
train_set_std = standardize(train_set_interaction)
train_set_final = insert_1(train_set_std)

test_set_interaction = interaction(test_set)
test_set_std = standardize(test_set_interaction)
test_set_final = insert_1(test_set_std)

def squared_error_loss(dff4,weight,sum_7yr):
    
    dff4 = dff4.as_matrix()
    mult = np.matmul(dff4,weight)
    error=np.subtract(mult,sum_7yr)
    error_sqr= np.power(error,2)
    sum_sqr_error=np.sum(error_sqr)/2
    return sum_sqr_error

def regularization(dff5,lamb,sum_7yr_gp):
    dff5=dff5.as_matrix()
    
    sum_7yr_gp=sum_7yr_gp.as_matrix(); 
       
    identity=np.identity(245)
    dff_transpose=np.transpose(dff5)
    
    a=(lamb*identity)+np.matmul(dff_transpose,dff5)
    c=np.matmul(dff_transpose,sum_7yr_gp)
    
    optimal_weight_vector=np.matmul(np.linalg.pinv(a),c)
    
    return optimal_weight_vector
    
def get_datasets(dff6,sum_7yr,iteration,k):
    end = 0
    dff6=pd.concat([dff6,sum_7yr], axis=1)
    size = dff6.shape[0]
    beg = int(np.ceil(size/k)*iteration)
    if(iteration <9):
        end =int(np.ceil(size/k)*(iteration+1))
        
    else:
        if(iteration==9) :
            end = 638
    testing = dff6.iloc[beg:end,]
    training = dff6.iloc[:beg,].append(dff6.iloc[end:,])
    testing_sum_7yr_gp=testing["sum_7yr_GP"]
    training_sum_7yr_gp=training["sum_7yr_GP"]
    testing = testing.drop("sum_7yr_GP", axis = 1)
    training = training.drop("sum_7yr_GP", axis = 1)
    return training, testing, training_sum_7yr_gp, testing_sum_7yr_gp
    
error_list_cv=[]
k=10
lamb_list = [0, 0.01, 0.1, 1, 10, 100, 1000]
for i in lamb_list:
    sum_error=0
    for j in range(k):
        
        training_lamb,test_lamb,training_sum_7yr_gp,testing_sum_7yr_gp = get_datasets(train_set_final,sum_7yr_gp_train,j,k)
        weight_vec = regularization(training_lamb,i,training_sum_7yr_gp)
        error = squared_error_loss(test_lamb,weight_vec,testing_sum_7yr_gp)
        sum_error=sum_error+error
    
    mean_error=sum_error/k
    error_list_cv.append(mean_error)

error_list_wo_cv=[]
for i in lamb_list:
    weight_wo_cv = regularization(train_set_final,i, sum_7yr_gp_train)
    error_wo_cv = squared_error_loss(test_set_final,weight_wo_cv,sum_7yr_gp_test)
    error_list_wo_cv.append(error_wo_cv)

def graph(validationError,label_name):
    dict_list = zip(validationError,lamb_list)
    d=dict(dict_list)
    best_val_error=min(validationError)
    best_lambda=d[best_val_error]
    plt.semilogx(lamb_list, validationError,label= label_name)
    plt.semilogx(best_lambda,best_val_error,marker='o',color='b',label="Best Lambda")
    plt.ylabel('Sum Squared Error')
    plt.text(5, 116, lamb_list, fontsize=15)
    #plt.text(5, 10, validationError, fontsize=15)
    plt.legend()
    plt.xlabel('Lambda')
    plt.show()
    
graph(error_list_wo_cv,"Training set")
graph(error_list_cv,"Testing set")