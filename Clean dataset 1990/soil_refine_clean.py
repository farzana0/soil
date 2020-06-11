#soil_refine_clean.py


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
from sklearn.feature_selection import SelectKBest, f_classif
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


sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white')


def plot_features(df, columns):
    missing = []
    for i in columns:
        print('number of missing data ' + i + ' :', (len(df[i]) - df[i].count()))
        missing.append(len(df[i]) - df[i].count())
        fig = plt.figure(i)
        
        scaler = MinMaxScaler()
        scaler.fit((df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1))
        X = scaler.transform((df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1)) 
        ax = sns.distplot(X, hist = True, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = i)
        ax2 = ax.twinx()
        sns.boxplot(x=X, ax=ax2)
        ax2.set(ylim=(-.5, 10))

        plt.ylabel('Density')
        plt.title(i)
        plt.savefig('0_' + i + '.png' )
        plt.close()
    dff = pd.DataFrame(dict(features=columns, number_of_missing=missing))
    sns.barplot(x='features', y = 'number_of_missing', data=dff)
    plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light') 
    plt.title('#missing data')
    plt.savefig('missing')
    plt.close()


def reform_targets(response_columns, df):
    for index, value in df['TEXTURE'].items():
        #print(value.split(' '))
        #print(index)
        df['TEXTURE'].update(pd.Series([value.split(' ')[0]], index=[index]))
    print(df['TEXTURE'].unique())
    print(df['TEXTURE'].value_counts())


def plot_y(y,df):
    plt.close()
    df['TEXTURE'].value_counts().plot.bar()
    plt.tight_layout()
    plt.savefig('plot_y')
    plt.close()


def feature_in_onefigure(data, target, df):
    fig,axes =plt.subplots(9,3, figsize=(12, 9)) # 3 columns each containing 9 figures, total 27 features
    ARGILE=df.loc[df['TEXTURE']=='ARGILE', data].to_numpy()
    SABLE=df[df.columns.intersection(data)].loc[df['TEXTURE']=='SABLE', data].to_numpy()
    LOAM=df[df.columns.intersection(data)].loc[df['TEXTURE']=='LOAM', data].to_numpy()

    ax=axes.ravel() # flat axes with numpy ravel
    for i in range(len(data)):
      _,bins=np.histogram(df[df.columns.intersection(data)].to_numpy()[:,i],bins=40)
      ax[i].hist(ARGILE[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
      ax[i].hist(SABLE[:,i],bins=bins,color='g',alpha=0.3)# alpha is for transparency in the overlapped region 
      ax[i].hist(LOAM[:,i],bins=bins,color='b',alpha=0.1)
      ax[i].set_title(df.columns.intersection(data)[i],fontsize=9)
      ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
      ax[i].set_yticks(())
    ax[0].legend(['ARGILE','SABLE', 'LOAM'],loc='best',fontsize=8)
    plt.tight_layout() # let's make good plots
    plt.savefig('feature')
    plt.close()


def feature(data, target, df):
    ARGILE=df.loc[df['TEXTURE']=='ARGILE', data].to_numpy()
    SABLE=df[df.columns.intersection(data)].loc[df['TEXTURE']=='SABLE', data].to_numpy()
    LOAM=df[df.columns.intersection(data)].loc[df['TEXTURE']=='LOAM', data].to_numpy()

    for i in range(len(data)):
        fig,ax =plt.subplots(1,1, figsize=(24, 14))
        _,bins=np.histogram(df[df.columns.intersection(data)].to_numpy()[:,i],bins=40)
        ax.hist(ARGILE[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
        ax.hist(SABLE[:,i],bins=bins,color='g',alpha=0.3)# alpha is for transparency in the overlapped region 
        ax.hist(LOAM[:,i],bins=bins,color='b',alpha=0.1)
        ax.set_title(df.columns.intersection(data)[i],fontsize=25)
        ax.set_yticks(())
        ax.legend(['ARGILE','SABLE', 'LOAM'],loc='best',fontsize=22)
        plt.tight_layout() # let's make good plots
        plt.savefig('feature' + str(i) + df.columns.intersection(data)[i] )
        plt.close()


def visualization2d(X, y):
    X = preprocessing.scale(X)
    data_subset = X
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    sns.set_palette("Set2")
    plt.figure(figsize=(16,10))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y
                         ,cmap='Set2', legend='brief') # hue = y
    plt.legend(title='Tsne', loc='upper left', labels=['ARGILE', 'LIMON', 'LOAM', 'SABLE'])
    plt.title('Tsne Visualization in 2D')
    plt.tight_layout()
    plt.savefig('Tsne')
    plt.close()

def visualization3d(X, y):
    X = preprocessing.scale(X)
    data_subset = X
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    ax = plt.axes(projection='3d')
    for  color, i, target_name in zip(mpl.cm.Set2.colors[:4], ['ARGILE', 'LIMON', 'LOAM', 'SABLE'], ['ARGILE', 'LIMON', 'LOAM', 'SABLE']):
        ax.scatter(tsne_results[np.where(y.to_numpy() == i), 0], tsne_results[np.where(y.to_numpy() == i), 1], tsne_results[np.where(y.to_numpy() == i), 2], 
             label=target_name, color=color)
    plt.title('tsne visualization' + " of chemical dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.tight_layout()
    plt.savefig('3d_tsne')
    plt.close()







#visualization(df[columns].astype('float'), df['TEXTURE'])
def pca(X):
    pca = PCA(2)
    pca.fit(X)
    print('number of the components', pca.n_components_ )
    print('explained variance', pca.explained_variance_)
    return(pca.fit_transform(X))

def clustering_dbscan(X, labels_true):
    X = StandardScaler().fit_transform(X)
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    # Plot result
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', #, markerfacecolor=tuple(col)
                  markeredgecolor='k' ,markersize=8. ) #markeredgecolor='k',

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', #, markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=2)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.legend()
    plt.savefig('clusteringdbscan')
    plt.close()


def clustering_kmeans(X, labels_true):

    X = StandardScaler().fit_transform(X)
    # #############################################################################
    # Compute DBSCAN
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    labels = kmeans.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    # Plot result
    # Black removed and is used for noise instead.
    #X = pca(X)
    #unique_labels = set(labels)
    sns.set_palette('Set2')
    sns.scatterplot(x=X[:, 0], y=X[:, 1],
                        hue=labels_true, style=labels, legend='brief')
    plt.savefig('clustering_kmeans')
    plt.close()



def feature_selection(X, y, data, number_features):
    bestfeatures = SelectKBest(score_func=chi2, k=number_features)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Chemicals','Score']  #naming the dataframe columns
    print(featureScores.nlargest(number_features,'Score'))  #print 10 best features
    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(number_features).plot(kind='barh')
    plt.rcParams.update({'font.size': 8})
    plt.title('Feature selection ExtraTreesClassifier')
    plt.savefig('feature_selection')
    plt.close()

    #get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(25,25))
    #plot heat map

    plt.rcParams.update({'font.size': 20})
    g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn", )
    plt.savefig('heatmap')
    plt.close()



def incremental_pca(X, y, data):
# Authors: Kyle Kastner
# License: BSD 3 clause
    n_components = 2
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)
    X_ipca = ipca.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        #plt.figure()
        for color, i, target_name in zip(mpl.cm.Set2.colors[:4] , ['ARGILE', 'LIMON', 'LOAM', 'SABLE'], ['ARGILE', 'LIMON', 'LOAM', 'SABLE']):
            plt.scatter(X_transformed[np.where(y.to_numpy() == i), 0], X_transformed[np.where(y.to_numpy() == i), 1],
                        color=color, label=target_name)

        if "Incremental" in title:
            err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            plt.title(title + " of chemical dataset\nMean absolute unsigned error "
                      "%.6f" % err)
        else:
            plt.title(title + " of chemical dataset")
        plt.legend(loc="best", shadow=False, scatterpoints=1)

        plt.savefig('2d_pca' + title)
        plt.close()

    n_components = 3
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)
    X_ipca = ipca.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    ax = plt.axes(projection='3d')
    for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
        #plt.figure()
        for color, i, target_name in zip(mpl.cm.Set2.colors[:4], ['ARGILE', 'LIMON', 'LOAM', 'SABLE'], ['ARGILE', 'LIMON', 'LOAM', 'SABLE']):
            ax.scatter(X_transformed[np.where(y.to_numpy() == i), 0], X_transformed[np.where(y.to_numpy() == i), 1], X_transformed[np.where(y.to_numpy() == i), 2],
                       color=color, label=target_name)
        if "Incremental" in title:
            err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
            plt.title(title + " of chemical dataset\nMean absolute unsigned error "
                     "%.6f" % err)
        else:
            plt.title(title + " of chemical dataset")
        plt.legend(loc="best", shadow=False, scatterpoints=1)

        plt.savefig('3d_pca' + title)
        plt.close()




def logistic_regression(X, y):
    print(mc.averageprec(X, y))



def xgboost(X, y):
    datamatrix  = xgb.DMatrix(data=X, label=y)
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
    xg_reg = OneVsRestClassifier(xgb.XGBClassifier())
    params = {"objective": 'multi:softprob', "eval_metric": 'merror','colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10, "num_class" :y.shape[1]}
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    y_score = xg_reg.predict_proba(X_test)
    accuracy = accuracy_score(y_test, preds)
    print("accuracy: %f", (accuracy))
    metrics_precision_recall(y_test, preds, X_test, n_classes, 'XGboost')
    #cv_results = xgb.cv(dtrain=datamatrix, params=params, nfold=10,
    #               num_boost_round=50,early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
    #print(cv_results)
    #print((cv_results["test-rmse-mean"]).tail(1))
    xg_reg = xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
    xgb.plot_tree(xg_reg,num_trees=0)
    plt.rcParams['figure.figsize'] = [70, 70]
    plt.rcParams.update({'font.size': 6})
    plt.savefig('tree.png')
    plt.close()
    #plt.show()
    plt.rcParams.update({'font.size': 65})
    xgb.plot_importance(xg_reg, max_num_features=10)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.savefig('xgboost')
    plt.close()


def onehotencoder(X):
    ohe = OneHotEncoder() 
    X_ohe = ohe.fit_transform(X) # It returns an numpy array
    return X_ohe.toarray()


def svmclassifier(X, Y):
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    classifier = LinearSVC(random_state=0, tol=1e-3)
    #Kfold cross-validation
    n_classes = 4
    kf = KFold(n_splits=10, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    print(kf) 
    accuracy = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        classifier.fit(X_train, Y_train)
        y_score = classifier.decision_function(X_test)
        preds = classifier.predict(X_test)
        accuracy.append(accuracy_score(Y_test, preds))

    accuracy_mean = statistics.mean(accuracy)
    accuracy_std = statistics.stdev(accuracy)
    print('svm:', accuracy_mean, accuracy_std)
    #import pdb
    #pdb.set_trace()
    classifier = LinearSVC(random_state=0, tol=1e-5)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,  random_state = 123)
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(X_test)
    metrics_precision_recall(onehotencoder(y_test.reshape(-1,1)), y_score, X_test, n_classes, 'SVM')
        
    

def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data


def dimensionality_reduction_f_classify(X, y):
    # Split dataset to select feature and evaluate the classifier
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=0
    )

    plt.figure(1)
    plt.rcParams['figure.figsize'] = [50, 50]
    plt.rcParams.update({'font.size': 6})
    plt.clf()
    X_indices = np.arange(X.shape[-1])
    # #############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function to select the four
    # most significant features
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X_train, y_train)
    scores = -np.log10(replaceZeroes(selector.pvalues_))
    scores /= scores.max()
    #print(scores)
    plt.rcParams['figure.figsize'] = [70, 70]
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)',
            edgecolor='black')

    # #############################################################################
    # Compare to the weights of an SVM
    clf = make_pipeline(MinMaxScaler(), LinearSVC())
    clf.fit(X_train, y_train)
    print('Classification accuracy without selecting features: {:.3f}'
          .format(clf.score(X_test, y_test)))

    svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()

    plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
             edgecolor='black')

    clf_selected = make_pipeline(
            SelectKBest(f_classif, k=5), MinMaxScaler(), LinearSVC()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after univariate feature selection: {:.3f}'
          .format(clf_selected.score(X_test, y_test)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
            width=.2, label='SVM weights after selection',
            edgecolor='black')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    #plt.axis('tight')
    #plt.legend(loc='upper right')
    
    plt.legend(loc='upper left', prop={'size': 8})
    plt.savefig('dim_reduction_f_classify')
    plt.close()



def compare_methods(X, y):
    pipe = Pipeline([
    # the reduce_dim stage is populated by the param_grid
    ('reduce_dim', 'passthrough'),
    ('classify', LinearSVC(dual=False, max_iter=10000))
    ])

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
    grid.fit(X, y)
    mean_scores = np.array(grid.cv_results_['mean_test_score'])
    # scores are in the order of param_grid iteration, which is alphabetical
    mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
    # select score for best C
    mean_scores = mean_scores.max(axis=0)
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                   (len(reducer_labels) + 1) + .5)
    plt.figure()
    #plt.rcParams['figure.figsize'] = [100, 100]
    plt.rcParams.update({'font.size': 25})
    plt.rcParams['figure.figsize'] = [50, 50]
    #plt.rcParams.update({'font.size': 10})
    #plot.legend(loc=2, prop={'size': 6})
    COLORS = 'bgrcmyk'
    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
        plt.bar(bar_offsets + i, reducer_scores, label=label)

    plt.title("Comparing feature reduction techniques")
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left', prop={'size': 20})
    plt.savefig('compare_methods')
    plt.close()



def Linear_Regression(X, Y):
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The coefficients
        #print('Coefficients: \n', regr.coef_)
        coefficients = coefficients + regr.coef_

        # The mean squared error
        #print('Mean squared error: %.2f'
         #     % mean_squared_error(y_test, y_pred))
        # The coefficient of determination: 1 is perfect prediction
        meansquared_error.append(mean_squared_error(y_test, y_pred))
        #print('Coefficient of determination: %.2f'
          #    % r2_score(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))

        #plot outputs
        '''mpl.rcParams.update(mpl.rcParamsDefault)
        plt.figure()
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(X_test)
        plt.scatter(X_pca, y_test,  color='black')
        plt.plot(X_pca, y_pred, color=mpl.cm.Set2.colors[0], linewidth=3)
        #plt.plot(np.unique(X_test[:,i]), np.poly1d(np.polyfit(X_test[:,i], y_pred, 1))(np.unique(X_test[:,i])))
        plt.xticks(())
        plt.yticks(())
        plt.show()
        plt.close()'''
    print('Coefficients: \n', np.multiply(coefficients,0.2))
    # The mean squared error
    print('Mean squared error: %.2f'
              % statistics.mean(meansquared_error))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
              % statistics.mean(r2score))



def svreg(X, Y):
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create Decision Tree regression object
           # Fit regression model
        regr= SVR(C=1.0, epsilon=0.2)

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The mean squared error

        # The coefficient of determination: 1 is perfect prediction
        meansquared_error.append(mean_squared_error(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))

    # The mean squared error
    print('Mean squared error SVR: %.2f'
              % statistics.mean(meansquared_error))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination SVR: %.2f'
              % statistics.mean(r2score))

def Knnn(X, Y):
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create Decision Tree regression object
           # Fit regression model
        regr= KNeighborsRegressor()

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The mean squared error

        # The coefficient of determination: 1 is perfect prediction
        meansquared_error.append(mean_squared_error(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))

    # The mean squared error
    print('Mean squared error Knn: %.2f'
              % statistics.mean(meansquared_error))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination Knn: %.2f'
              % statistics.mean(r2score))

def extratree(X, Y):
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create Decision Tree regression object
           # Fit regression model
        regr= ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
                                       random_state=0)

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The mean squared error

        # The coefficient of determination: 1 is perfect prediction
        meansquared_error.append(mean_squared_error(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))

    # The mean squared error
    print('Mean squared error ExtraTreesRegressor: %.2f'
              % statistics.mean(meansquared_error))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination ExtraTreesRegressor: %.2f'
              % statistics.mean(r2score))





def DFR(X, Y):
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create Decision Tree regression object
           # Fit regression model
        regr= DecisionTreeRegressor(max_depth=4)

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The mean squared error

        # The coefficient of determination: 1 is perfect prediction
        meansquared_error.append(mean_squared_error(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))

    # The mean squared error
    print('Mean squared error DF: %.2f'
              % statistics.mean(meansquared_error))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination DF: %.2f'
              % statistics.mean(r2score))





def ADFR(X, Y):
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    rng = np.random.RandomState(1)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # Create Decision Tree regression object
           # Fit regression model
        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(X_test)

        # The mean squared error
        # The coefficient of determination: 1 is perfect prediction
        meansquared_error.append(mean_squared_error(y_test, y_pred))
        #print('Coefficient of determination: %.2f'
          #    % r2_score(y_test, y_pred))
        r2score.append(r2_score(y_test, y_pred))
            # The mean squared error
    print('Mean squared error ADF: %.2f'
              % statistics.mean(meansquared_error))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination ADF: %.2f'
              % statistics.mean(r2score))

def Xgboost_regression(X, y, param):
    datamatrix  = xgb.DMatrix(data=X, label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    params = {"objective": 'reg:squarederror', "eval_metric": 'rmse', 'colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10}
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2score = r2_score(y_test, preds)
    print("RMSE: %f" % (rmse))
    print("r2score: %f" % (r2score))
    cv_results = xgb.cv(dtrain=datamatrix, params=params, nfold=5,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    print(cv_results.head())
    print((cv_results["test-rmse-mean"]).tail(1))
    xg_reg = xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
    xgb.plot_tree(xg_reg, num_trees=0)
    plt.rcParams['figure.figsize'] = [70, 70]
    plt.rcParams.update({'font.size': 6})
    plt.savefig('tree_regression' + param + '.png')
    plt.close()
    #plt.show()
    plt.rcParams.update({'font.size': 65})
    xgb.plot_importance(xg_reg, max_num_features=10)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.savefig('xgboost_regression' + param)
    plt.close()

def multioutputregression(X, Y):
    # Fit estimators
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
                                       random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    rng = np.random.RandomState(1)
    meansquared_error_es = dict()
    r2score_es = dict ()
    for name, estimator in ESTIMATORS.items():
        meansquared_error = []
        r2score = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            meansquared_error.append(mean_squared_error(y_test, y_pred))
            r2score.append(r2_score(y_test, y_pred))
        meansquared_error_es[name] = statistics.mean(meansquared_error)
        r2score_es[name] = statistics.mean(r2score)
    print(meansquared_error_es)
    print(r2score_es)


def chainregressor(X, Y):
    # Fit estimators
    ESTIMATORS = {
        "Extra trees + chain": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
                                       random_state=0),
        "K-nn + chain": KNeighborsRegressor(),
        "Linear regression + chain": LinearRegression(),
        "Ridge + chain": RidgeCV(),
    }
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    rng = np.random.RandomState(1)
    meansquared_error_es = dict()
    r2score_es = dict ()
    for name, estimator in ESTIMATORS.items():
        meansquared_error = []
        r2score = []
        estimator = RegressorChain(estimator)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            meansquared_error.append(mean_squared_error(y_test, y_pred))
            r2score.append(r2_score(y_test, y_pred))
        meansquared_error_es[name] = statistics.mean(meansquared_error)
        r2score_es[name] = statistics.mean(r2score)
    print(meansquared_error_es)
    print(r2score_es)   

MultiOutputRegressor

def multioutputregression_wrapper(X, Y):
    # Fit estimators
    ESTIMATORS = {
        "Extra trees + wrapper": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
                                       random_state=0),
        "K-nn +  wrapper": KNeighborsRegressor(),
        "Linear regression + wrapper ": LinearRegression(),
        "Ridge + wrapper": RidgeCV(),
    }
    kf = KFold(n_splits=5, shuffle=True) # Define the split - into
    kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    accuracy = []
    r2score = []
    meansquared_error = []
    coefficients = 0
    rng = np.random.RandomState(1)
    meansquared_error_es = dict()
    r2score_es = dict ()
    for name, estimator in ESTIMATORS.items():
        meansquared_error = []
        r2score = []
        estimator = MultiOutputRegressor(estimator)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            meansquared_error.append(mean_squared_error(y_test, y_pred))
            r2score.append(r2_score(y_test, y_pred))
        meansquared_error_es[name] = statistics.mean(meansquared_error)
        r2score_es[name] = statistics.mean(r2score)
    print(meansquared_error_es)
    print(r2score_es)



        
def neauralnetwork_classification(X, y):
    nn_nn.classification(X, y)   


def neauralnetwork_regression(X, y):
    nr.regression(X, y)  




def read_data(columns, response_columns):
	 df = pd.read_excel('Tabi_ML.xlsx', usecols = columns + response_columns)
	 return df

def summary_of_data(df, columns, response_columns):

	pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print(df.describe())
    print('number of feature:', len(columns))
    print('number of targets:', len(response_columns))
    print('labels:', df[response_columns].unique())
    print('number of rows:', df.shape[0])
    print(df.head(3))
    plot_features(df, columns)
    print('labels distribution:')
    print(df[response_columns].value_counts(normalize=True) * 100)

    df = df.dropna(inplace=True)

    #aggregate the targets from 13 rto 4 groups
    reform_targets(response_columns, df)
    print('labels distribution aggregated:')

    print(df[response_columns].value_counts(normalize=True) * 100)
    
    plot_y(response_columns, df)
    feature(columns, response_columns, df)

def clustering(df, columns, response_columns):

	clustering_dbscan(pca(df[columns].to_numpy()), df[response_columns])      
    clustering_kmeans(pca(df[columns].to_numpy()), df[response_columns])
    pca(df[columns])

def encoded_labels(df, response_columns):
	response_column = df[response_columns]

    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(response_column)
    label_encoded_y = label_encoder.transform(response_column)

    df.drop(response_columns[0], axis = 1, inplace = True)
    # Put whatever series you want in its place
    df['TEXTURE'] = label_encoded_y

def visualization(df, columns, response_columns):

	visualization2d(df[columns], df[response_columns])
    visualization3d(df[columns], df[response_columns])
    incremental_pca(df[columns], df[response_columns], df)

def feature_selection(df, columns, response_columns):
	feature_selection(df[columns], df['TEXTURE'], df, 5)


def one_hot_encoder(df, columns, response_columns):
    a = np.array(df['TEXTURE'].to_numpy().reshape(1, df['TEXTURE'].to_numpy().shape[0]))
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


def compare_methods_dim_reduction(df, columns, response_columns)
	dimensionality_reduction_f_classify(df[columns].to_numpy() , df[response_columns].to_numpy())
    compare_methods(df[columns].to_numpy() , df[response_columns].to_numpy())

def describe_targets(df, response_columns):


def classification():
    columns = []
    response_columns= []  
    #response_columns = ['SABLETG',	'SABLEG',	'SABLEM',	'SABLEF',	'SABLETF',	'SABLE'	,'LIMONF',	'LIMONM',	'LIMONG',	'LIMON'	,'ARGILE',	'BATTANCE'	,'CONDHYD',	'ML',	'MN']
    response_columns = ['TEXTURE']
    columns = ['MO',	'N'	,'INDICE20'	,'PHEAU', 'PHSMP',	'KECH', 'CAECH'	, 'MGECH',	'NAECH'	, 'HECH', 'CEC', 'PM3',	'MNM3',	'CUM3'	,'FEM3'	,'ALM3'	,'BM3',	'ZNM3',	'PBM3',	'MOM3',	'CDM3',	'COM3',	'CRM3'	,'KM3'	,'CAM3'	,'MGM3',	'NAM3']  
    columns_ECH = columns[:11]
    columns_M3 = columns[:5] + columns[11:]
    #component = ['B', 'Co', 'Mn', 'Cu', 'P', 'K', 'Fe', 'Al', 'Cr', 'Na', 'Ca', 'Mo', 'Mg']
    columns = columns_M3

    df = read_data(columns, response_columns)
    summary_of_data(df, columns, response_columns)
    visualization(df, columns, response_columns)
    clustering(df, columns, response_columns)
	encoded_labels(df, response_columns)
	feature_selection(df, columns, response_columns)

    b = one_hot_encoder(df, columns, response_columns)

    # CLASSIFICATION
    neauralnetwork_classification(df[columns].to_numpy() ,b)
    logistic_regression(df[columns] , b)
    svmclassifier(df[columns].to_numpy() ,df[response_columns].to_numpy())
    xgboost(df[columns] , b)


    compare_methods_dim_reduction(df, columns, response_columns)




def prediction():
    columns = []
    response_columns= []

    #SABLE
    #LIMON
    #ARGILE

    response_columns = ['SABLE', 'LIMON', 'ARGILE']
    columns = ['MO',    'N' ,'INDICE20' ,'PHEAU', 'PHSMP',  'KECH', 'CAECH' , 'MGECH',  'NAECH' , 'HECH', 'CEC', 'PM3', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3', 'ZNM3', 'PBM3', 'MOM3', 'CDM3', 'COM3', 'CRM3'  ,'KM3'  ,'CAM3' ,'MGM3',    'NAM3']
    columns_M3 = columns[1:5] + [columns[9]] + columns[11:] 
    coulmns = columns_M3

    df = read_data(columns, response_columns)

    summary_of_data(df, columns, response_columns)
	

   

    # predict each target independently

    for response in response_columns:
        print('mean of the data:', statistics.mean(np.multiply(0.01,df[response].to_numpy()))) #percentage
        print('standard deviation of the data:', statistics.stdev(np.multiply(0.01,df[response].to_numpy()))) # percentage
        print(Xgboost_regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy()), response))
        print(Linear_Regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(DFR(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(ADFR(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(Knnn(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(extratree(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(svreg(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(neauralnetwork_regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
        print(nrp.regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))


    multioutputregression_wrapper(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy())
    chainregressor(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy()))
    multioutputregression(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy()))
    print(mnr.regression(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy())))





classification()
prediction()


