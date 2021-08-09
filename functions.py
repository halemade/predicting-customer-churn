#import the appropriate tools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, recall_score, precision_score,\
accuracy_score,f1_score,confusion_matrix,plot_confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
import time
warnings.filterwarnings('ignore')

def vanilla_models(X,y,test_size=.3):
    """ This function takes in predictors, a target variable and an optional test
    size parameter and returns results for 9 baseline classifiers"""
    
    names = ["Logistic Regression","Nearest Neighbors","Naive Bayes", "Linear SVM", "RBF SVM","Decision Tree",
             "Random Forest", "Gradient Boost", "AdaBoost","XGBoost"]
    
    req_scaling = ["Nearest Neighbors"]

    classifiers = [
        LogisticRegression(),
        KNeighborsClassifier(3),
        GaussianNB(),
        SVC(kernel="linear", C=.5),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        XGBClassifier()
        ]  
    
    #init df to hold report info for all classifiers
    df = pd.DataFrame(columns = ['classifier','train accuracy','train precision',
                                 'train recall','train f1 score','test accuracy',
                                 'test precision','test recall','test f1 score',
                                 'test time'])
    
    #train test splitsies
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3,random_state=42)
    
    #iterate over classifiers
    for count,clf in enumerate(classifiers):
        start = time.time()
        scaler = StandardScaler()
        if names[count] in req_scaling:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)
 
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        clf.fit(X_train_scaled,y_train)
        train_preds = clf.predict(X_train_scaled)
        test_preds = clf.predict(X_test_scaled)
        
        #training stats
        train_recall = round(recall_score(y_train,train_preds,average = 'weighted'),3)
        train_precision = round(precision_score(y_train,train_preds,average='weighted'),3)
        train_acc = round(accuracy_score(y_train,train_preds),3)
        train_f1 = round(f1_score(y_train,train_preds,average='weighted'),3)
        
        #testing stats
        recall = round(recall_score(y_test,test_preds,average='weighted'),3)
        precision = round(precision_score(y_test,test_preds,average='weighted'),3)
        f1 = round(f1_score(y_test,test_preds,average='weighted'),3)
        cm = confusion_matrix(y_test,test_preds)
        acc = round(accuracy_score(y_test,test_preds),3)
        end = time.time()
        elapsed = round((end-start),2)
        
        #append results to dataframe
        df = df.append({'classifier':names[count],'train accuracy':train_acc,
                        'train precision':train_precision,'train recall':train_recall,
                        'train f1 score':train_f1,'test accuracy':acc,
                        'test precision':precision,'test recall':recall,
                        'test f1 score':f1,'test time':elapsed},ignore_index=True)
    return df

def run_model(clf,X,y):
    #train test splitsies
    """takes in an instantiated classifier and the predictive and target data. 
    use only for models on that do not require data scaling"""
    
    start = time.time()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
    X_train, y_train = SMOTE().fit_resample(X_train,y_train)
    clf.fit(X_train,y_train)
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    model_report = classification_report(y_test, test_preds,target_names = labels.keys(),output_dict = True)

    #training stats
    train_recall = round(recall_score(y_train,train_preds,average = 'weighted'),3)
    train_precision = round(precision_score(y_train,train_preds,average='weighted'),3)
    train_acc = round(accuracy_score(y_train,train_preds),3)
    train_f1 = round(f1_score(y_train,train_preds,average='weighted'),3)

    #testing stats
    recall = round(recall_score(y_test,test_preds,average='weighted'),3)
    precision = round(precision_score(y_test,test_preds,average='weighted'),3)
    f1 = round(f1_score(y_test,test_preds,average='weighted'),3)
    cm = confusion_matrix(y_test,test_preds)
    acc = round(accuracy_score(y_test,test_preds),3)
    end = time.time()
    elapsed = round((end-start),2)
    #append results to dataframe
    report = dict({'classifier':clf,'train accuracy':train_acc,
                    'train precision':train_precision,'train recall':train_recall,
                    'train f1 score':train_f1,'test accuracy':acc,
                    'test precision':precision,'test recall':recall,
                    'test f1 score':f1,'test time':elapsed})
    #plot confusion matrix
    train_plot = plot_confusion_matrix(clf,X_train,y_train)
    test_plot = plot_confusion_matrix(clf,X_test,y_test)
    return report, "Top plot: Training Data", "Bottom Plot: Testing Data"


def run_scaled_model(clf,X,y):
    #train test splitsies
    """takes in an instantiated classifier and the predictive and target data. 
    use only for models on that require data scaling"""
    start = time.time()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)
    X_train, y_train = SMOTE().fit_resample(X_train,y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train,y_train)
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
   

    #training stats
    train_recall = round(recall_score(y_train,train_preds,average = 'weighted'),3)
    train_precision = round(precision_score(y_train,train_preds,average='weighted'),3)
    train_acc = round(accuracy_score(y_train,train_preds),3)
    train_f1 = round(f1_score(y_train,train_preds,average='weighted'),3)

    #testing stats
    recall = round(recall_score(y_test,test_preds,average='weighted'),3)
    precision = round(precision_score(y_test,test_preds,average='weighted'),3)
    f1 = round(f1_score(y_test,test_preds,average='weighted'),3)
    cm = confusion_matrix(y_test,test_preds)
    acc = round(accuracy_score(y_test,test_preds),3)
    end = time.time()
    elapsed = round((end-start),2)
    #append results to dataframe
    report = dict({'classifier':clf,'train accuracy':train_acc,
                    'train precision':train_precision,'train recall':train_recall,
                    'train f1 score':train_f1,'test accuracy':acc,
                    'test precision':precision,'test recall':recall,
                    'test f1 score':f1,'test time':elapsed})
    #plot confusion matrix
    train_plot = plot_confusion_matrix(clf,X_train,y_train)
    test_plot = plot_confusion_matrix(clf,X_test,y_test)
    return report, "Top plot: Training Data", "Bottom Plot: Testing Data"

def plot_importances(model_dict,X):
    features = dict(zip(X.columns,model_dict[0]['classifier'].feature_importances_))
    fi = pd.DataFrame({
    "features": list(X.columns),
    "importances": model[0]['classifier'].feature_importances_ ,
    })
    sort = fi.sort_values(by=['features'])
    fig = px.bar(sort, x="features", y="importances", barmode="group")
    fig.update_layout(title = 'XGBoost Feature Importances')    
    return fig.show()
