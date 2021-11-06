from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
x_train = pickle.load(open('x_train.pkl', 'rb'))
model = pickle.load(open('list_of_all_models.pkl', 'rb'))

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


le1 = LabelEncoder()
le = le1.fit(x_train)
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
        NewClientName1 = le.transform(NewClientName)
        NewProductName1 = le.transform(NewProductName)
        Town1 = le.transform(Town)
        State1 = le.transform(State)
        brand1 = le.transform(brand)
        prediction=model.predict([[WeekNumber,SalesDepotID,SalesChannelID,RouteID,ClientID,ProductID,NewClientName1,NewProductName1,pieces,weight,brand1,Town1,State1]])
        output=round(prediction[0])
        if output<0:
            return render_template('index.html',prediction_texts="There is no demand for this particular product")
        else:
            return render_template('index.html',prediction_text="The Demand for the product is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

