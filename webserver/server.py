from flask import Flask, render_template, request, url_for
import sqlite3
from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo
import pandas as pd
import joblib
craw=pd.DataFrame
rraw=pd.DataFrame

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
jl_filepath=jl_filedir / 'reg_obesity.joblib'
rf3_jl=joblib.load(jl_filepath)


app=Flask(__name__)



@app.route("/")
def welcome():
        return render_template('welcome.html')

@app.route("/about")
def about():
        return render_template('about.html')

@app.route("/regressor")
def regressor():
        return render_template('regressor.html')

@app.route("/classifier")
def classifier():
        return render_template('classifier.html')

@app.route("/data/<model>")
def data(model):
        if(model=='regression'):
                con=sqlite3.connect('./database/models.db')
                cur1=con.cursor();cur2=con.cursor()
                sample=rraw.sample(5)
                # raw=cur1.execute("select * from rawData order by random() limit 5")
                clean=cur2.execute("select * from reg_clean_data order by random() limit 5")
                return render_template('data_reg.html',raw=sample,clean=clean,varDesc=rvarDesc)
        if(model=='classification'):
                con=sqlite3.connect('./database/models.db')
                cur1=con.cursor();cur2=con.cursor()
                sample=craw.sample(5)
                # raw=cur1.execute("select * from rawData order by random() limit 5")
                clean=cur2.execute("select * from class_clean_data order by random() limit 5")
                return render_template('data_class.html',raw=sample,clean=clean,varDesc=cvarDesc)

if __name__=="__main__":
        app.run(debug=True)

