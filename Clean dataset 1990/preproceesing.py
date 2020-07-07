#preproceesing.py
import numpy as np 
import csv
import pandas as pd 
import glob
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
def regression(X,y, x):
	reg = LinearRegression().fit(X, y)
	print(reg.score(X, y))
	print(reg.coef_)
	return reg.predict(x)
def maketensor():
	path = "*.csv"
	dflist = []
	for name in sorted(glob.glob(path)):
		#print(pd.read_csv(name, engine='python').describe())
		dflist.append(pd.read_csv(name, engine='python'))
		print(len(dflist))
		print(name)
def variableovertime(col):
	path = "*.csv"
	dflist = []
	df = pd.DataFrame({'DESCODPR1' : []})
	for name in sorted(glob.glob(path)):
		#print(pd.read_csv(name, engine='python').head())
		#print(name.split('_'))
		#df1 = pd.DataFrame(pd.read_csv(name, engine='python')['SUP']).rename(columns={'SUP': name.split('_')[2]}, inplace=True)
		df1 = pd.read_csv(name, engine='python', usecols=[col])#.rename(columns={'SUP': name.split('_')[2]}, inplace=True)
		#print(df.head())
		#print(df1.head())
		dflist.append(df1.iloc[:,0].str.lower())
		df = df1.join(df, lsuffix = name.split('_')[2])
		#print(df.head())
		#print(len(dflist))
		#print(name)
	df.drop(df.columns[-1],axis=1,inplace=True)
	combined = pd.concat(dflist, ignore_index=True)
	print(combined.head())
	print(combined.shape)
	print(combined.value_counts()[:30])
	return df
def makenumpymatrix():
	path = "*.csv"
	dflist = []
	for name in sorted(glob.glob(path)):
		my_data = genfromtxt(name, delimiter=',')
		dflist.append(my_data)
	return np.array(dflist)
def plotsummaryoffeatures():
	df = pd.read_csv('BDPPAD_v03_2003_s_20161026.csv', engine='python')
	for i in df.columns:
		try:
			fig = plt.figure()
			#fig.title(i)
			plt.hist(df[i])
			plt.show()
		except:
			pass  
def culture(df):
	#print(df.groupby(axis=1).count())
	list_dict = []
	for column in reversed(df.columns):
		#print(df[column].nunique)
		list_dict.append(df[column].value_counts().to_dict())
		#fig = plt.figure()
		#fig.title(i)
		#plt.hist(df[column])
		#plt.show()
#maketensor()
#print(np.shape(makenumpymatrix()))
#plotsummaryoffeatures()
print(variableovertime(16).head())
#print(culture(variableovertime(16)))