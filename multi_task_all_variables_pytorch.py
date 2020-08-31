# multi_task_all_variables_pytorch.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import pandas as pd
import math
import sklearn.preprocessing as sk
from tensorboardX import SummaryWriter
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score
import torch.utils.data as Data
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler
from statistics import mean


def normalizedata(data):
    num_features = data.shape[1]

    mean = np.array([data[:,j].mean() for j in range(num_features)]).reshape(num_features)
    std = np.array([data[:,j].std() for j in range(num_features)]).reshape(num_features)

    for i in range(num_features):
        if float(std[i]) != 0:
            data[:, i] = (data[:, i] - float(mean[i])) * (1 / float(std[i]))
        else:
            data[:, i] = np.ones((data.shape[0]))
    return data


class MTLnet(nn.Module):
    def __init__(self):
        super(MTLnet, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU()
            # nn.Dropout()
        )
        # self.sharedlayer_1 = nn.Sequential(
        #     nn.Linear(shared_layer_size_1, shared_layer_size),
        #     nn.ReLU()
        #     # nn.Dropout()
        # )
        # self.sharedlayer_2 = nn.Sequential(
        #     nn.Linear(shared_layer_size_1, shared_layer_size),
        #     nn.ReLU()
        #     # nn.Dropout()
        # )
        self.shared_tower1 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h2, tower_h3),
            nn.ReLU()
        )
        self.tower1_1 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )   
        self.tower1_2 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.tower1_3 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.shared_tower2 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h2, tower_h3),
            nn.ReLU()
             )      
        self.tower2_1 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )   
        self.tower2_2 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.tower2_3 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.shared_tower3 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h2, tower_h3),
            nn.ReLU()

             )   

        self.tower3_1 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )   
        self.tower3_2 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.tower3_3 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.shared_tower4 = nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h2, tower_h3),
            nn.ReLU()
             ) 
        self.tower4_1 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )   
        self.tower4_2 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        self.tower4_3 = nn.Sequential(
            nn.Linear(tower_h3, tower_h4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h4, tower_h5),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(tower_h2, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(tower_h5, output_size)
        )
        # self.shared_tower5 = nn.Sequential(
        #     nn.Linear(shared_layer_size, tower_h1),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h2, tower_h3),
        #     nn.ReLU()
        #      )  
        # self.tower5_1 = nn.Sequential(
        #     nn.Linear(tower_h3, tower_h4),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h4, tower_h5),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h5, output_size)
        # )
        # self.tower5_2 = nn.Sequential(
        #     nn.Linear(tower_h3, tower_h4),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h4, tower_h5),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h5, output_size)
        # )
        # self.tower5_3 = nn.Sequential(
        #     nn.Linear(tower_h3, tower_h4),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h4, tower_h5),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h5, output_size)
        # )


        # self.shared_tower6 = nn.Sequential(
        #     nn.Linear(shared_layer_size, tower_h1),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h1, tower_h2),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h2, tower_h3),
        #     nn.ReLU()
        #      )  
        # self.tower6_1 = nn.Sequential(
        #     nn.Linear(tower_h3, tower_h4),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h4, tower_h5),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h5, output_size)
        # )
        # self.tower6_2 = nn.Sequential(
        #     nn.Linear(tower_h3, tower_h4),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h4, tower_h5),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h5, output_size)
        # )
        # self.tower6_3 = nn.Sequential(
        #     nn.Linear(tower_h3, tower_h4),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h4, tower_h5),
        #     nn.ReLU(),
        #     # nn.Dropout(),
        #     # nn.Linear(tower_h2, tower_h2),
        #     # nn.ReLU(),
        #     # nn.Dropout(),
        #     nn.Linear(tower_h5, output_size)
        # )
    def forward(self, x):
        h_shared = self.sharedlayer(x)

        # h_shared_1 = self.sharedlayer_1(h_shared)
        # h_shared_2 = self.sharedlayer_2(h_shared)


        out1 = self.shared_tower1(h_shared)
        out2 = self.shared_tower2(h_shared)
        out3 = self.shared_tower3(h_shared)
        out4 = self.shared_tower4(h_shared)
        # out5 = self.shared_tower5(h_shared)
        # out6 = self.shared_tower6(h_shared)

        out1_1 = self.tower1_1(out1)
        out1_2 = self.tower1_2(out1)
        out1_3 = self.tower1_3(out1)

        out2_1 = self.tower1_1(out2)
        out2_2 = self.tower1_2(out2)
        out2_3 = self.tower1_3(out2)

        out3_1 = self.tower1_1(out3)
        out3_2 = self.tower1_2(out3)
        out3_3 = self.tower1_3(out3)

        out4_1 = self.tower1_1(out4)
        out4_2 = self.tower1_2(out4)
        out4_3 = self.tower1_3(out4)

        # out5_1 = self.shared_tower1_1(out5)
        # out5_2 = self.shared_tower1_1(out5)
        # out5_3 = self.shared_tower1_1(out5)

        # out6_1 = self.shared_tower1_1(out6)
        # out6_2 = self.shared_tower1_1(out6)
        # out6_3 = self.shared_tower1_1(out6)
        return [out1_1, out1_2, out1_3, out2_1, out2_2, out2_3, out3_1, out3_2, out3_3, out4_1, out4_2, out4_3]



loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
def loss_function_multi(prediction, y):
    loss = []
    ls = 0
    for ye in range(y.shape[1]):
        loss.append(loss_func(prediction[ye], y[:,ye]))
        ls = ls + loss_func(prediction[ye], y[:,ye])
    return loss, (ls / 12)



loss = 0
def train(x, y, x_valid, y_valid, xtest, ytest):
    BATCH_SIZE = 50
    EPOCH = 500
    net = MTLnet()
    # x, y = Variable(torch.from_numpy(x).type(torch.FloatTensor)), Variable(torch.from_numpy(y).type(torch.FloatTensor))
    torch_dataset = Data.TensorDataset(x, y)

    learning_rate = 0.1
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)


    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # start training

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step

            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            # b_y = b_y.view(b_y.size()[0], 1)

            prediction = net(b_x)     # input x and predict based on x



            loss_list, loss = loss_function_multi(prediction, b_y)    # must be (1. nn output, 2. target)



            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        print('train Loss at step {0}: {1}'.format(epoch, loss))
        if epoch % 100 == 0:      
            with torch.no_grad():
                yhat = net(x_valid)
                a,losshat =loss_function_multi(yhat, y_valid)
                print('validation Loss at step {0}: {1}'.format(epoch, losshat))
        exp_lr_scheduler.step()



    # xtest, ytest = Variable(torch.from_numpy(xtest).type(torch.FloatTensor)), Variable(torch.from_numpy(ytest).type(torch.FloatTensor))
    # ytest = ytest.view(ytest.size()[0], 1)
    predictionhat = net(xtest)     # input x and predict based on x
    a,loss = loss_function_multi(predictionhat, ytest)
    a,loss_test = loss_function_multi(predictionhat, ytest)
    print('test loss {0}'.format(loss_test)) 
    for i in range(ytest.shape[1]):
        print('test loss {0}'.format(a[i]))
    for i in range(ytest.shape[1]):
        print('test r2score {0}'.format(r2_score(ytest[:,i].data.numpy(), predictionhat[i].data.numpy())))








columns = []
response_columns= []

#SABLE
#LIMON
#ARGILE

# response_columns = ['SABLE', 'LIMON', 'ARGILE']
# response_columns = ['MVA', 'CONDHYD', 'DMP', 'POROTOT', 'MO']
response_columns = ['MVA', 'POROTOT1', 'PORODRAI1', 'CH_cm_h' ] #DMP
# response_columns=['PCMO']
columns = ['PM3', 'CEC', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3'  ,'KM3'  ,'CAM3' ,'MGM3', 'ARGILE', 'SABLE', 'LIMON', 'CentreEp', 'PHSMP', 'PHEAU']


df1 = pd.read_csv('Couche_Inv1990tot.csv', usecols = ['IDEN2.x'] + ['IDEN3'] + ['GROUPE.x'] + ['Couche'] + columns + response_columns, encoding='latin-1')
df2 = pd.read_csv('Site_Inv1990.csv', usecols = ['IDEN2'] + ['xcoord', 'ycoord'], encoding='latin-1')
df3 = pd.read_csv('Champ_Inv1990.csv', usecols = ['IDEN3', 'Culture_1']  , encoding='latin-1')
# print(df1.columns)
# print(df2.columns)
# print(df3.columns)


# df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
df4 = df1.merge(df2, left_on='IDEN2.x', right_on='IDEN2').reindex(columns=['IDEN2', 'IDEN3', 'GROUPE.x', 'Couche' ] + ['xcoord', 'ycoord'] + columns + response_columns)
df = df4.merge(df3, left_on='IDEN3', right_on='IDEN3')
# geopandas_(df)

# print('stop')
# print(df.columns)
# plot_features(df, columns, 'Alltogether')


# print(df['Culture_1'].unique())
# print(df['Culture_1'].value_counts())


# use pd.concat to join the new columns with your original dataframe
df = pd.concat([df,pd.get_dummies(df['Culture_1'])],axis=1)

# now drop the original 'country' column (you don't need it anymore)
df.drop(['Culture_1'],axis=1, inplace=True)
# print(df.columns)

columns = columns + ['xcoord', 'ycoord'] + ['1-prairie',
   '2-céréales', '3-maïs-grain', '4-pommes de terre', '5-maïs-ensilage',
   '6-autres']

from functools import reduce

for column in columns + response_columns:
	df[column].fillna((df[column].mean()), inplace=True)


cc = columns.copy()

df.dropna(inplace = True, subset=response_columns)
dfcouche1 = df[df['Couche'] == 1]
dfcouche2 = df[df['Couche'] == 2]
dfcouche3 = df[df['Couche'] == 3]


dfff = [0, 0 , 0]




print(dfcouche1['IDEN2'].count())
print(dfcouche1['IDEN2'].nunique())

print(dfcouche2['IDEN2'].count())
print(dfcouche2['IDEN2'].nunique())

print(dfcouche3['IDEN2'].count())

YY = []
for response in response_columns:

    dfff[0] =  dfcouche1
    dfff[1] =  dfcouche2[['IDEN2', response]]
    dfff[2] =  dfcouche3[['IDEN2', response]]
    print(dfcouche3[response].nunique())

    df_ = reduce(lambda  left,right: pd.merge(left,right,on=['IDEN2'],how='inner'), dfff)
    # df_ = dfcouche1.merge(dfcouche2[['IDEN2', 'MVA']], 'inner', on=['IDEN2'], suffixes=['_1', '_2'])
    # df_ = df_.merge(dfcouche3['IDEN2', 'MVA'], 'inner', on=['IDEN2'], suffixes=['_3'])
    df_.rename ({response + '_x': response + '_1', response + '_y': response + '_2', response : response + '_3'}, axis=1, inplace=True)

    print(df_.columns)
    # MVA = [df_['MVA_1'], df_['MVA_2'], df_['MVA_3']]

    # Y = np.array[([dfcouche1['MVA'], dfcouche2['MVA']])]
    Y1 = np.array([df_[response + '_1']]).flatten()
    Y2 = np.array([df_[response + '_2']]).flatten()
    Y3 = np.array([df_[response + '_3']]).flatten()

    YY.append(Y1)
    YY.append(Y2)
    YY.append(Y3)


    #YY is appending all the above

X = df_[columns].to_numpy()

X = normalizedata(X)


YY=np.array([np.array(xi) for xi in YY]).transpose()





# YY = normalizedata(YY)
print(X.shape)
print(YY.shape)
# print(YY.columns)



L = Y2.shape[0]
L1 = L - 100
L2 = L - 200
N = L

split = list(np.random.permutation(N))

X_train = X[split[0:L2],:]

YY_train = YY[split[0:L2], :]


X_valid = X[L2:L1,:]
YY_valid = YY[L2:L1, :]




X_test = X[L1:L,:]
YY_test = YY[L1:L,:]



X_train = torch.from_numpy(X_train)
X_train = X_train.float()

YY_train =  torch.from_numpy(YY_train)
YY_train =  YY_train.float()


X_valid = torch.from_numpy(X_valid)
X_valid = X_valid.float()

YY_valid = torch.from_numpy(YY_valid)
YY_valid = YY_valid.float()



X_test = torch.from_numpy(X_test)
X_test = X_test.float()

YY_test = torch.from_numpy(YY_test)
YY_test = YY_test.float()  


print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)




input_size, feature_size = X.shape
shared_layer_size_1 = 1024
shared_layer_size = 512
tower_h1 = 256
tower_h2 = 128
tower_h3 = 64
tower_h4 = 32
tower_h5 = 16

output_size = 1
LR = 0.01
epoch = 500
mb_size = 10




# l1D = loss_func(Yhat1, Y1_test.view(-1,1))
# l2D = loss_func(Yhat2, Y2_test.view(-1,1))
# l3D = loss_func(Yhat3, Y3_test.view(-1,1))



# print(r2_score(Y1_test.view(-1,1), Yhat1.data.numpy()))
# print(r2_score(Y2_test.view(-1,1), Yhat2.data.numpy()))
# print(r2_score(Y3_test.view(-1,1), Yhat3.data.numpy()))



# plt.plot(np.squeeze(costtr)[-100:], '-r',np.squeeze(costD)[-100:], '-b')
# plt.ylabel('total cost')
# plt.xlabel('iterations (per tens)')
# plt.show() 

# plt.plot(np.squeeze(cost1tr)[-50:], '-r', np.squeeze(cost1D)[-50:], '-b')
# plt.ylabel('task 1 cost')
# plt.xlabel('iterations (per tens)')
# plt.show() 

# plt.plot(np.squeeze(cost2tr)[-50:],'-r', np.squeeze(cost2D)[-50:],'-b')
# plt.ylabel('task 2 cost')
# plt.xlabel('iterations (per tens)')
# plt.show()


train(X_valid, YY_valid, X_valid, YY_valid, X_valid, YY_valid)