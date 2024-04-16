# WORK WITH FILES
from ucimlrepo import fetch_ucirepo # pulls datasets from uci ML respository
import sqlite3
import csv
import joblib
from pathlib import Path
# DATA MANIPULATION
from rfpimp import *
import category_encoders as ce
import pandas as pd
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_numeric_dtype
# GENERAL STUFF
import warnings
warnings.filterwarnings('ignore')
# LIBRARIES TO WORK WITH MODELS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
# metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

# ------------------------------------------- HELPER FUNCTIONS ----------------------------------------------

# ------------------------------------------- METRICS FUNCTIONS ----------------------------------------------
def cross_val(model,X,y,type):
        kfold=KFold(n_splits=20,random_state=7, shuffle=True)

        if type=='c':
            results=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
        else:
            if type=='r':
                results=cross_val_score(model,X,y,cv=kfold,scoring='r2')

        print(f'-------------Cross Validation-----------------')
        print(f'Accuracy -val set: {results.mean()*100}')
        # # splitting into train and validation
        Xtrain,Xval,ytrain,yval=train_test_split(X,y, test_size=0.24, random_state=7)
        model.fit(Xtrain,ytrain)
        result=model.score(Xval,yval)
        print(f'Accuracy -test set: {result*100.0}')



# ------------------------------------------- EVALUATE FUNCTIONS ---------------------------------------------
# -- create models and evaluate them
def evalReg(X,y,dt):
    oob_scores = []
    for i in range(10):
        rf = RandomForestRegressor(n_estimators=dt, n_jobs=-1, oob_score=True)
        rf.fit(X, y)
        oob_scores.append(rf.oob_score_)
    oob=sum(oob_scores) / len(oob_scores)
    print(f'Mean OOB score: {oob}')
    print(f'{rfnnodes(rf):,d} tree nodes and {np.median(rfmaxdepths(rf))} median tree height')
    return rf, oob

def evalClass(X,y,dt):
    oob_scores = []
    for i in range(10):
        rf = RandomForestClassifier(n_estimators=dt, n_jobs=-1, oob_score=True)
        rf.fit(X, y)
        oob_scores.append(rf.oob_score_)
    oob=sum(oob_scores) / len(oob_scores)
    print(f'Mean OOB score: {oob}')
    print(f'{rfnnodes(rf):,d} tree nodes and {np.median(rfmaxdepths(rf))} median tree height')
    return rf, oob
# ----------------------------------------- FEATURES IMPORTANCES --------------------------------------------

def showimp(rf,X,y):
    I=importances(rf,X,y)
    plot_importances(I,color='blue')
    return I
# -------------------------------------------- READING DATA FUNCTIONS ----------------------------------------
# --- we took this function from book, useful to quickly identify most obvious missing values in dataset ---
def sniff(df):
    # with pd.option_context("display.max_colwidth",20):
        info=pd.DataFrame()
        #info['sample']=df.iloc[120] #no needed here
        info['data type']=df.dtypes
        info['percent missing']=df.isnull().sum()*100/len(df)
        info['No. unique'] = df.apply(lambda x: len(x.unique()))
        info['unique values'] = df.apply(lambda x: x.unique())
        return info.sort_values('data type')

# ---------------- we'll use this many times during this lab, so we're saving keystrokes -------------------
def get_unique_values_in_columns(df):
     print('PRINTING UNIQUE VALUES PER COLUMN\n')
     for col in df.columns:
        print(f'{col} : {df[col].value_counts()} {df[col].unique()}')
        #  print(f'{col} : ')

#--------------------------------------- FUNCTIONS TO NORMALIZE --------------------------------------------
#                           missing values in numeric and non-numeric features
# NON-NUMERIC FEATURES: receives dataframe to normalize and a string with all values to replace
# and returns it 
def normalize_nonum(df,missing_values_str):
    for col in df.columns:
        df[col]=df[col].str.lower()
        df[col]=df[col].fillna(np.nan)
        df[col]=df[col].replace(missing_values_str, np.nan)
    return df
# SET ALL VALUES IN NON-NUMERIC FEATURES TO LOWERCASE
def to_lower(df):   
    for col in df.columns:
        if is_object_dtype(df[col]):
            df[col]=df[col].str.lower()
    return df

# NUMERIC FEATURES: receives the dataframe to normalize and returns it
def normalize_num(df,mv):
    for col in df.columns:
        df[col]=df[col].replace(int(mv),np.nan)
    return df

# --------------------------------------- FUNCTIONS TO LABEL ENCODE -----------------------------------------
#                non-numeric features as ordinal, taken from book, this will fix missing values
# this converts dtype for each column to ordinal category.
def df_string_to_cat(df):
    for col in df.columns:
        if is_object_dtype(df[col]):
            df[col] = df[col].astype('category').cat.as_ordered()
    return df
# this loop transform nominal values in to integers according to categories previously assigned
def df_cat_to_catcode(df):
    for col in df.columns:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes + 1
    return df

# ------------------------- FUNCTION TO HANDLE MISSING VALUES ---------------------------
#                in numeric features based on the one showed in the book.
def fix_missing_num(df):
    r=df.isna().any()  # get the columns with missing values
    msvrows=r[r==True].index
    for e in msvrows:       #now loop through those columns and replace missing values with median
        df[e+'_na']=pd.isnull(df[e])
        df[e].fillna(df[e].median(), inplace=True)
        