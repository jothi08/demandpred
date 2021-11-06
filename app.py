from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
model = pickle.load(open('list_of_all_models.pkl', 'rb'))

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
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
        NewClientName = LabelEncoder().fit_transform(NewClientName)
        NewProductName = LabelEncoder().fit_transform(NewProductName)
        Town = LabelEncoder().fit_transform(Town)
        State = LabelEncoder().fit_transform(State)
        brand = LabelEncoder().fit_transform(brand)
        prediction=model.predict([[WeekNumber,SalesDepotID,SalesChannelID,RouteID,ClientID,ProductID,NewClientName,NewProductName,pieces,weight,brand,Town,State]])
        print("prediction:",prediction)
        output=round(prediction)
        if output<0:
            return render_template('index.html',prediction_texts="There is no demand for this particular product")
        else:
            return render_template('index.html',prediction_text="The Demand for the product is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

