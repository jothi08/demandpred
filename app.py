from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
import joblib
import Final as final

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        WeekNumber = int(request.form['WeekNumber'])
        SalesDepotID=int(request.form['SalesDepotID'])
        SalesChannelID=int(request.form['SalesChannelID'])
        RouteID=int(request.form['RouteID'])
        ClientID=int(request.form['ClientID'])
        ProductID=int(request.form['ProductID'])
        NewClientName=request.form['NewClientName']
        NewProductName=request.form['NewProductName']
        pieces=int(request.form['pieces'])
        weight=int(request.form['weight'])
        brand=request.form['brand']
        Town=request.form['Town']
        State=request.form['State']
        p=np.array([[WeekNumber,SalesDepotID,SalesChannelID,RouteID,ClientID,ProductID,NewClientName,NewProductName,pieces,weight,brand,Town,State]])
        d1=pd.DataFrame(p,columns =['WeekNumber','SalesDepotID','SalesChannelID','RouteID','ClientID','ProductID','NewClientName','NewProductName','pieces','weight','brand','Town','State'])
        prediction = final.final_f1(d1)
        
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="There is no demand for this particular product")
        else:
            return render_template('index.html',prediction_text="The Demand for the product is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
