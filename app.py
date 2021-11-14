from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
model = pickle.load(open('model1_3.pkl', 'rb'))
Mean_encoded_SalesDepotID= pickle.load(open('Mean_encoded_SalesDepotID.pkl','rb'))
Mean_encoded_SalesChannelID= pickle.load(open('Mean_encoded_SalesChannelID.pkl','rb'))
Mean_encoded_RouteID= pickle.load(open('Mean_encoded_RouteID.pkl','rb'))
Mean_encoded_ClientID= pickle.load(open('Mean_encoded_ClientID.pkl','rb'))
Mean_encoded_ProductID= pickle.load(open('Mean_encoded_ProductID.pkl','rb'))
leNewClientName_map= pickle.load(open('leNewClientName_map.pkl', 'rb'))
leNewProductName_map= pickle.load(open('leNewProductName_map.pkl', 'rb'))
leTown_map= pickle.load(open('leTown_map.pkl', 'rb'))
leState_map= pickle.load(open('leState_map.pkl', 'rb'))
lebrand_map= pickle.load(open('lebrand_map.pkl', 'rb'))

app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


le1 = LabelEncoder()
st1 = StandardScaler()
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
        piece=int(request.form['piece'])
        weight=int(request.form['weight'])
        brand=request.form['brand']
        Town=request.form['Town']
        State=request.form['State']
        piece1=np.log(piece+1)
        weight1=np.log(weight+1)
        NewClientName1 = leNewClientName_map.get(NewClientName)
        NewProductName1 = leNewProductName_map.get(NewProductName)
        brand1 = lebrand_map.get(brand)
        Town1 = leTown_map.get(Town)
        State1 = leState_map.get(State)
        SalesDepotID1=Mean_encoded_SalesDepotID.get(SalesDepotID)
        SalesChannelID1=Mean_encoded_SalesChannelID.get(SalesChannelID)
        RouteID1=Mean_encoded_RouteID.get(RouteID)
        ClientID1=Mean_encoded_ClientID.get(ClientID)
        ProductID1=Mean_encoded_ProductID.get(ProductID)
        p=np.array([[WeekNumber,SalesDepotID1,SalesChannelID1,RouteID1,ClientID1,ProductID1,NewClientName1,NewProductName1,piece1,weight1,brand1,Town1,State1]])
        d1=pd.DataFrame(p)
        prediction=[]
        for i in range(15):
            prediction1=model[i].predict(d1)
            output1=round(np.exp(prediction1[0]))
            prediction.append(output1)
        
        output=int(np.mean(np.array(prediction)))
        if output<0:
            return render_template('index.html',prediction_texts="There is no demand for this particular product")
        else:
            return render_template('index.html',prediction_text="The Demand for the product is {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

