# tsne and pca embeddings

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import linear_model
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, NMF
import multilabelClassification as mc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from multilabelClassificationMetrics import metrics_precision_recall
import NN as nn_nn
import NN_regression as nr
import matplotlib as mpl
from sklearn.isotonic import IsotonicRegression
import statistics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
import NN_regression_pytorch as nrp
import multioutput_nn_regressor as mnr
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor
import uuid
import matplotlib as mpl
from imputer import impute
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from mpl_toolkits.mplot3d import Axes3D

def incremental_pca(X, y,  group, name):
# Authors: Kyle Kastner
# License: BSD 3 clause
	n_components = 2
	ipca = IncrementalPCA(n_components=n_components, batch_size=10)
	X_ipca = ipca.fit_transform(X)

	pca = PCA(n_components=n_components)
	X_pca = pca.fit_transform(X)
	for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
		#plt.figure()
		for color, i, target_name in zip(mpl.cm.Set2.colors[:len(group)] , group, group):
			plt.scatter(X_transformed[np.where(y.to_numpy() == i), 0], X_transformed[np.where(y.to_numpy() == i), 1],
						color=color, label=target_name)

		if "Incremental" in title:
			err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
			plt.title(title + " of chemical dataset\nMean absolute unsigned error "
					  "%.6f" % err)
		else:
			plt.title(title + " of chemical dataset")
		plt.legend(loc="best", shadow=False, scatterpoints=1)

		plt.savefig('2d_pca' + title + ' ' + name)
		plt.close()

	n_components = 3
	ipca = IncrementalPCA(n_components=n_components, batch_size=10)
	X_ipca = ipca.fit_transform(X)

	pca = PCA(n_components=n_components)
	X_pca = pca.fit_transform(X)

	ax = plt.axes(projection='3d')
	for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
		#plt.figure()
		for color, i, target_name in zip(mpl.cm.Set2.colors[:len(group)], group, group):
			ax.scatter(X_transformed[np.where(y.to_numpy() == i), 0], X_transformed[np.where(y.to_numpy() == i), 1], X_transformed[np.where(y.to_numpy() == i), 2],
						color=color, label=target_name)
		if "Incremental" in title:
			err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
			plt.title(title + " of chemical dataset\nMean absolute unsigned error "
					 "%.6f" % err)
		else:
			plt.title(title + " of chemical dataset")
		plt.legend(loc="best", shadow=False, scatterpoints=1)

		plt.savefig('3d_pca' + title + ' ' + name)
		plt.close()





def visualization2d(X,y, group, name):
	X = preprocessing.scale(X)
	data_subset = X
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(data_subset)
	sns.set_palette("Set2")
	plt.figure(figsize=(16,10))
	sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y
						 ,cmap='Set2', legend='brief') # hue = y
	plt.legend(title='Tsne', loc='upper left', labels=group)
	plt.title('Tsne Visualization in 2D')
	plt.tight_layout()
	plt.savefig('Tsne' + ' ' + name)
	plt.close()



def visualization3d(X, y, group, name):
	X = preprocessing.scale(X)
	data_subset = X
	tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(data_subset)
	ax = plt.axes(projection='3d')
	for  color, i, target_name in zip(mpl.cm.Set2.colors[:len(group)], group, group):
		ax.scatter(tsne_results[np.where(y.to_numpy() == i), 0], tsne_results[np.where(y.to_numpy() == i), 1], tsne_results[np.where(y.to_numpy() == i), 2], 
			 label=target_name, color=color)
	plt.title('tsne visualization' + " of chemical dataset")
	plt.legend(loc="best", shadow=False, scatterpoints=1)
	plt.tight_layout()
	plt.savefig('3d_tsne'+ ' ' + name)
	plt.close()



def main(X, y, group, name):
	incremental_pca(X,y,group, name)
	visualization2d(X,y,group, name)
	visualization3d(X,y,group, name)




