from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
model = pickle.load(open('list_of_all_models.pkl', 'rb'))

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


le1 = LabelEncoder()
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
        NewClientName1 = le1.fit_transform(['NewClientName'])
        NewProductName1 = le1.fit_transform(['NewProductName'])
        brand1 = le1.fit_transform(['brand'])
        Town1 = le1.fit_transform(['Town'])
        State1 = le1.fit_transform(['State'])
        p=np.array([[WeekNumber,SalesDepotID,SalesChannelID,RouteID,ClientID,ProductID,NewClientName1[0],NewProductName1[0],pieces,weight,brand1[0],Town1[0],State1[0]]])
        d1=pd.DataFrame(p)
        prediction=model[4].predict(d1)
        output=round(np.exp(prediction[0]))
        if output<0:
            return render_template('index.html',prediction_texts="There is no demand for this particular product")
        else:
            return render_template('index.html',prediction_text="The Demand for the product is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

