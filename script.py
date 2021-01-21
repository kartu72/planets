import pandas as pd
import shap
import sys
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocessing(data: pd.DataFrame):
    f = 'class' in data
    if f:
        labels=data['class']
        data=data.drop(columns='class')
    data=data.apply(pd.to_numeric)
    data=data.drop(columns=['objid', 'ra', 'dec'])
    #data=data.drop(columns=['objid'])
    if f:
        return (data, labels)
    else:
        return data

def null_processing(data: pd.DataFrame)  ->  pd.DataFrame:
    data=data.replace({'na':np.NaN})
    data.fillna(data.median(), inplace=True)
    return data

def null_knn(data: pd.DataFrame)  ->  pd.DataFrame:
    data=data.replace({'na':np.NaN})
    imputer = KNNImputer(n_neighbors=30,metric='nan_euclidean')
    data = pd.DataFrame(imputer.fit_transform(data),columns = data.columns)
    return data

def predict_info(predict):
    a0=0
    a1=0
    a2=0
    for j in predict:
        if j==0:
            a0+=1
        elif j==1:
            a1+=1
        elif j==2:
            a2+=1
    print(a0,a1,a2)

def Clustering1(data: pd.DataFrame, train_data: pd.DataFrame)->pd.DataFrame:
    for band in ['u','g','r','i','z']:
        columns = []
        for i in range(6):
            s=band+'_'+str(i)
            columns.append(s)
        cluster_name='cluster_'+band
        model = KMeans(n_clusters = 10, random_state = 42)
        model.fit(train_data[columns])
        data[cluster_name]=model.predict(data[columns])
    return data

def Clustering2(data: pd.DataFrame, train_data: pd.DataFrame)->pd.DataFrame:
    for i in range(6):
        columns = []
        for band in ['u','g','r','i','z']:
            s=band+'_'+str(i)
            columns.append(s)
        cluster_name='cluster_'+str(i)
        model = KMeans(n_clusters = 10, random_state = 42)
        model.fit(train_data[columns])
        data[cluster_name]=model.predict(data[columns])
    return data

def main():
    args = sys.argv[1:]
    train_file = args[0]
    unlabeled_file = args[1]
    test_file = args[2]
    results_file = args[3]
    
    train_data=shuffle(pd.read_csv(train_file))
    unlabeled_data=shuffle(pd.read_csv(train_file))
    test_data=pd.read_csv(train_file)
    
    objid=test_data.objid
    train_data, train_labels = preprocessing(train_data)
    test_data = preprocessing(test_data)
    unlabeled_data = preprocessing(unlabeled_data)
    
    unlabeled_data=null_knn(unlabeled_data)
    train_data=null_knn(train_data)
    test_data=null_knn(test_data)
    
    train_data=Clustering1(train_data, pd.concat([train_data, unlabeled_data]))
    test_data=Clustering1(test_data, pd.concat([train_data, unlabeled_data]))
    unlabeled_data = Clustering1(unlabeled_data, pd.concat([train_data, unlabeled_data]))
    train_data=Clustering2(train_data, pd.concat([train_data, unlabeled_data]))
    test_data=Clustering2(test_data, pd.concat([train_data, unlabeled_data]))
    unlabeled_data = Clustering2(unlabeled_data, pd.concat([train_data, unlabeled_data]))
    
        
    params = { 
        'learning_rate': 0.025,
        'n_estimators': 1000,
        'max_depth': 6,
        'subsample': 0.75,
        'colsample_bytree': 0.5,
        'gamma': 1
    }
    model = XGBClassifier(**params, random_state=42)
    model.fit(train_data, train_labels)
    predictions=model.predict(test_data)
    out_df = pd.DataFrame({"objid": objid, "predictions": predictions})
    out_df.to_csv(results_file, index=False)
    
if __name__ == "__main__":
    main()

!ipython nbconvert — to script script.ipynb

