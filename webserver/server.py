from flask import Flask, render_template, request, url_for
import sqlite3
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo
import joblib

craw=pd.DataFrame()
rraw=pd.DataFrame()

obesity = fetch_ucirepo(id=544)
rraw=obesity.data.original
rvarDesc=obesity.variables

heartDisease = fetch_ucirepo(id=45)
cfeatures=heartDisease.data.features
ctarget=heartDisease.data.targets

craw=pd.concat([cfeatures,ctarget],axis=1)
cvarDesc=heartDisease.variables

jl_filedir = Path("./trained_models")
jl_filedir.mkdir(parents=True,exist_ok=True)




app=Flask(__name__)



@app.route("/")
def welcome():
        return render_template('welcome.html')

@app.route("/about")
def about():
        return render_template('about.html')

@app.route("/regressor", methods=['GET','POST']) # ------------------------- ROUTE TO LOAD REGRESSION MODEL
def regressor():
        jl_filepath=jl_filedir / 'reg_obesity.joblib'
        regressor=joblib.load(jl_filepath)
        data=[""]
        if request.method=='POST':
                arr=[]
                data=pd.DataFrame(columns=['Age','Height','FCVC','NCP','CH2O','FAF','TUE','Gender','family_history_with_overweight','FAVC','CAEC','SCC','CALC','MTRANS','Height_na'])
                for x in data.columns:
                  arr.append(request.form[x])
                data=regressor.predict([arr])
        return render_template('regressor.html', pred=data)

@app.route("/classifier", methods=['GET','POST']) # ---------------------------- ROUTE TO LOAD CLASSIFICATION MODEL
def classifier():
        jl_filepath=jl_filedir / 'class_heart.joblib'
        classifier=joblib.load(jl_filepath)
        data=""
        if request.method=='POST':
                arr=[]
                data=pd.DataFrame(columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','ca_na','thal_na'])
                for x in data.columns:
                  arr.append(request.form[x])
                data=classifier.predict([arr])
        return render_template('classifier.html', pred=data)

@app.route("/data/<model>")
def data(model):
        if(model=='regression'):
                con=sqlite3.connect('./database/models.db')
                cur1=con.cursor();cur2=con.cursor()
                sample=rraw.sample(5)
                clean=cur2.execute("select * from reg_clean_data order by random() limit 5")
                return render_template('data_reg.html',raw=sample,clean=clean,varDesc=rvarDesc)
        if(model=='classification'):
                con=sqlite3.connect('./database/models.db')
                cur1=con.cursor();cur2=con.cursor()
                sample=craw.sample(5)
                clean=cur2.execute("select * from class_clean_data order by random() limit 5")
                return render_template('data_class.html',raw=sample,clean=clean,varDesc=cvarDesc)

if __name__=="__main__":
        app.run(debug=True)

