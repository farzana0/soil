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



class MTLnet(nn.Module):
    def __init__(self):
        super(MTLnet, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            # nn.Dropout()
        )
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
        out1 = self.shared_tower1(h_shared)
        out2 = self.shared_tower2(h_shared)
        out3 = self.shared_tower3(h_shared)
        out4 = self.shared_tower4(h_shared)
        # out5 = self.shared_tower5(h_shared)
        # out6 = self.shared_tower6(h_shared)

        out1_1 = self.shared_tower1_1(out1)
        out1_2 = self.shared_tower1_1(out1)
        out1_3 = self.shared_tower1_1(out1)

        out2_1 = self.shared_tower1_1(out2)
        out2_2 = self.shared_tower1_1(out2)
        out2_3 = self.shared_tower1_1(out2)

        out3_1 = self.shared_tower1_1(out3)
        out3_2 = self.shared_tower1_1(out3)
        out3_3 = self.shared_tower1_1(out3)

        out4_1 = self.shared_tower1_1(out4)
        out4_2 = self.shared_tower1_1(out4)
        out4_3 = self.shared_tower1_1(out4)

        # out5_1 = self.shared_tower1_1(out5)
        # out5_2 = self.shared_tower1_1(out5)
        # out5_3 = self.shared_tower1_1(out5)

        # out6_1 = self.shared_tower1_1(out6)
        # out6_2 = self.shared_tower1_1(out6)
        # out6_3 = self.shared_tower1_1(out6)
        return [out1_1, out1_2, out1_3, out2_1, out2_2, out2_3, out3_1, out3_2, out3_3, out4_1, out4_2, out4_3]

def random_mini_batches(XE, R1E, R2E, R3E, mini_batch_size = 10, seed = 42): 
    # Creating the mini-batches
    np.random.seed(seed)            
    m = XE.shape[0]                  
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_XE = XE[permutation,:]
    shuffled_X1R = R1E[permutation]
    shuffled_X2R = R2E[permutation]
    shuffled_X3R = R3E[permutation]
    num_complete_minibatches = math.floor(m/mini_batch_size)


    for k in range(0, int(num_complete_minibatches)):
        mini_batch_XE = shuffled_XE[k * mini_batch_size : (k+1) * mini_batch_size, :]
        mini_batch_X1R = shuffled_X1R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X2R = shuffled_X2R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_X3R = shuffled_X3R[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R, mini_batch_X3R)
        mini_batches.append(mini_batch)
    
    Lower = int(num_complete_minibatches * mini_batch_size)
    Upper = int(m - (mini_batch_size * math.floor(m/mini_batch_size)))
    if m % mini_batch_size != 0:
        mini_batch_XE = shuffled_XE[Lower : Lower + Upper, :]
        mini_batch_X1R = shuffled_X1R[Lower : Lower + Upper]
        mini_batch_X2R = shuffled_X2R[Lower : Lower + Upper]
        mini_batch_X3R = shuffled_X3R[Lower : Lower + Upper]
        mini_batch = (mini_batch_XE, mini_batch_X1R, mini_batch_X2R, mini_batch_X3R)
        mini_batches.append(mini_batch)
    
    return mini_batches


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

columns = []
response_columns= []

#SABLE
#LIMON
#ARGILE

# response_columns = ['SABLE', 'LIMON', 'ARGILE']
# response_columns = ['MVA', 'CONDHYD', 'DMP', 'POROTOT', 'MO']
# response_columns = ['MVA', 'POROTOT1', 'POROTOT3', 'PORODRAI1', 'PORODRAI3', 'CH_cm_h' ] #DMP
response_columns=['PCMO']
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

for column in columns:
	df[column].fillna((df[column].mean()), inplace=True)


cc = columns.copy()


for response in response_columns:
    df.dropna(inplace = True, subset=[response])
    dfcouche1 = df[df['Couche'] == 1]
    dfcouche2 = df[df['Couche'] == 2]
    dfcouche3 = df[df['Couche'] == 3]


    dfff = [0, 0 , 0]


    dfff[0] =  dfcouche1
    dfff[1] =  dfcouche2[['IDEN2', response]]
    dfff[2] =  dfcouche3[['IDEN2', response]]


    print(dfcouche1['IDEN2'].count())
    print(dfcouche1['IDEN2'].nunique())

    print(dfcouche2['IDEN2'].count())
    print(dfcouche2['IDEN2'].nunique())

    print(dfcouche3['IDEN2'].count())
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

    Y4 = np.array([df_[response + '_1']]).flatten()
    Y5 = np.array([df_[response + '_2']]).flatten()
    Y6 = np.array([df_[response + '_3']]).flatten()

    Y7 = np.array([df_[response + '_1']]).flatten()
    Y8 = np.array([df_[response + '_2']]).flatten()
    Y9 = np.array([df_[response + '_3']]).flatten()

    Y10 = np.array([df_[response + '_1']]).flatten()
    Y11 = np.array([df_[response + '_2']]).flatten()
    Y12 = np.array([df_[response + '_3']]).flatten()

    #YY is appending all the above

    X = df_[columns].to_numpy()
    X = normalizedata(X)




    L = Y2.shape[0]
    L1 = L - 100
    L2 = L - 200
    N = L

    split = list(np.random.permutation(N))

    X_train = X[split[0:L2],:]

    YY_train = YY[split[0:L2], :]

    # Y1_train = Y1[split[0:L2]]
    # Y2_train = Y2[split[0:L2]]
    # Y3_train = Y3[split[0:L2]]

    # Y4_train = Y4[split[0:L2]]
    # Y5_train = Y5[split[0:L2]]
    # Y6_train = Y6[split[0:L2]]

    # Y7_train = Y7[split[0:L2]]
    # Y8_train = Y8[split[0:L2]]
    # Y9_train = Y9[split[0:L2]]

    # Y10_train = Y10[split[0:L2]]
    # Y11_train = Y11[split[0:L2]]
    # Y12_train = Y12[split[0:L2]]


    X_valid = X[L2:L1,:]
    YY_valid = YY[L2:L1, :]

    # Y1_valid = Y1[L2:L1]
    # Y2_valid = Y2[L2:L1]
    # Y3_valid = Y3[L2:L1]

    # Y4_valid = Y4[L2:L1]
    # Y5_valid = Y5[L2:L1]
    # Y6_valid = Y6[L2:L1]

    # Y7_valid = Y7[L2:L1]
    # Y8_valid = Y8[L2:L1]
    # Y9_valid = Y9[L2:L1]

    # Y10_valid = Y10[L2:L1]
    # Y11_valid = Y11[L2:L1]
    # Y12_valid = Y12[L2:L1]


    X_test = X[L1:L,:]
    YY_test = YY[L1:L,:]

    # Y1_test = Y1[L1:L]
    # Y2_test = Y2[L1:L]
    # Y3_test = Y3[L1:L]

    # Y4_test = Y4[L1:L]
    # Y5_test = Y5[L1:L]
    # Y6_test = Y6[L1:L]

    # Y7_test = Y7[L1:L]
    # Y8_test = Y8[L1:L]
    # Y9_test = Y9[L1:L]

    # Y10_test = Y10[L1:L]
    # Y11_test = Y11[L1:L]
    # Y12_test = Y12[L1:L]


    # print(X_train.shape)
    # print(X_valid.shape)
    # print(X_test.shape)
    # print(Y1_train.shape)
    # print(Y2_train.shape)
    # print(Y1_valid.shape)
    # print(Y2_valid.shape)
    # print(Y1_test.shape)
    # print(Y2_test.shape)

    X_train = torch.from_numpy(X_train)
    X_train = X_train.float()

    YY_train =  torch.from_numpy(YY_train)
    YY_train =  YY_train.float()

    # Y1_train = torch.tensor(Y1_train)
    # Y1_train = Y1_train.float()
    # Y2_train = torch.tensor(Y2_train)
    # Y2_train = Y2_train.float()
    # Y3_train = torch.tensor(Y3_train)
    # Y3_train = Y3_train.float()

    X_valid = torch.from_numpy(X_valid)
    X_valid = X_valid.float()

    YY_valid = torch.from_numpy(YY_valid)
    YY_valid = YY_valid.float()
    # Y1_valid = torch.tensor(Y1_valid)
    # Y1_valid = Y1_valid.float()
    # Y2_valid = torch.tensor(Y2_valid)
    # Y2_valid = Y2_valid.float()
    # Y3_valid = torch.tensor(Y3_valid)
    # Y3_valid = Y3_valid.float()


    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()

    YY_test = torch.from_numpy(YY_test)
    YY_test = YY_test.float()  
    # Y1_test = torch.tensor(Y1_test)
    # Y1_test = Y1_test.float()
    # Y2_test = torch.tensor(Y2_test)
    # Y2_test = Y2_test.float()
    # Y3_test = torch.tensor(Y3_test)
    # Y3_test = Y3_test.float()

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    # print(Y1_train.shape)
    # print(Y2_train.shape)
    # print(Y1_valid.shape)
    # print(Y2_valid.shape)
    # print(Y1_test.shape)
    # print(Y2_test.shape)



    input_size, feature_size = X.shape
    shared_layer_size = 64
    tower_h1 = 32
    tower_h2 = 16
    output_size = 1
    LR = 0.01
    epoch = 500
    mb_size = 10


    costtr = []
    # cost1tr = []
    # cost2tr = []
    # cost3tr = []

    # cost1D = []
    # cost2D = []
    # cost3D = []
    # cost1ts = []
    # cost2ts = []
    # cost3ts = []
    # costtr = []
    # costD = []
    # costts = []



    MTL = MTLnet()
    optimizer = torch.optim.Adam(MTL.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    for it in range(epoch):
        epoch_cost = 0
        epoch_cost1 = 0
        epoch_cost2 = 0
        epoch_cost3 = 0
        num_minibatches = int(input_size / mb_size) 
        minibatches = random_mini_batches(X_train, Y1_train, Y2_train, Y3_train, mb_size)
        

        for minibatch in minibatches:
            XE, YE1, YE2, YE3  = minibatch         
            Yhat1, Yhat2, Yhat3 = MTL(XE)

            l1 = loss_func(Yhat1, YE1.view(-1,1))    
            l2 = loss_func(Yhat2, YE2.view(-1,1))
            l3 = loss_func(Yhat3, YE3.view(-1,1))
            loss =  (l1 + l2 + l3)/3
            # loss = l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_cost = epoch_cost + (loss / num_minibatches)
            epoch_cost1 = epoch_cost1 + (l1 / num_minibatches)
            epoch_cost2 = epoch_cost2 + (l2 / num_minibatches)
            epoch_cost3 = epoch_cost3 + (l3 / num_minibatches)
        

        exp_lr_scheduler.step()
        costtr.append(torch.mean(epoch_cost))
        cost1tr.append(torch.mean(epoch_cost1))
        cost2tr.append(torch.mean(epoch_cost2))
        cost3tr.append(torch.mean(epoch_cost3))

        with torch.no_grad():
            Yhat1D, Yhat2D, Yhat3D = MTL(X_valid)
            l1D = loss_func(Yhat1D, Y1_valid.view(-1,1))
            l2D = loss_func(Yhat2D, Y2_valid.view(-1,1))
            l3D = loss_func(Yhat3D, Y3_valid.view(-1,1))
            cost1D.append(l1D)
            cost2D.append(l2D)
            cost3D.append(l3D)

            costD.append((l1D+l2D+l3D)/3)
            print('Iter-{}; Total loss: {:.4}'.format(it, (l1D+l2D+l3D)/3))
            print('Iter-{}; Total loss: {:.4}'.format(it, loss.data))
     

    Yhat1, Yhat2, Yhat3 = MTL(X_test)

    l1D = loss_func(Yhat1, Y1_test.view(-1,1))
    l2D = loss_func(Yhat2, Y2_test.view(-1,1))
    l3D = loss_func(Yhat3, Y3_test.view(-1,1))


    print(response)
    print(l1D)
    print(l2D)
    print(l3D)

    print(r2_score(Y1_test.view(-1,1), Yhat1.data.numpy()))
    print(r2_score(Y2_test.view(-1,1), Yhat2.data.numpy()))
    print(r2_score(Y3_test.view(-1,1), Yhat3.data.numpy()))



    plt.plot(np.squeeze(costtr)[-100:], '-r',np.squeeze(costD)[-100:], '-b')
    plt.ylabel('total cost')
    plt.xlabel('iterations (per tens)')
    plt.show() 

    plt.plot(np.squeeze(cost1tr)[-50:], '-r', np.squeeze(cost1D)[-50:], '-b')
    plt.ylabel('task 1 cost')
    plt.xlabel('iterations (per tens)')
    plt.show() 

    plt.plot(np.squeeze(cost2tr)[-50:],'-r', np.squeeze(cost2D)[-50:],'-b')
    plt.ylabel('task 2 cost')
    plt.xlabel('iterations (per tens)')
    plt.show()



