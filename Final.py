import numpy as np
import pandas as pd
#import dask.dataframe as dd
# https://www.kaggle.com/greentearus/bimbo-w-dask-and-call
final_types1 = {'WeekNumber': np.uint8,
                'NewClientName': np.int64,
                'NewProductName': np.float64,
                'pieces':np.float64,
                'weight':np.float64,
                'brand':np.int64,
                'Town':np.int64,
                'State':np.int64,
                'MeanR':np.float64,
                'MeanSC':np.float64,
                'MeanSD':np.float64,
                'MeanP':np.float64,
                'MeanC':np.float64,
                'MeanPSD':np.float64,
                'MeanPSC':np.float64,
                'MeanPR':np.float64,
                'MeanPC':np.float64,
                'MeanPCSC':np.float64,
                'MeanPCSD':np.float64,
                }

final_types = {'WeekNumber': np.uint8,
                'SalesDepotID': np.uint16,
                'SalesChannelID':np.uint8,
                'RouteID':np.uint16,
                'ClientID':np.uint32,
                'ProductID':np.uint16,
                'Demand':np.uint8,
                'NewClientName': np.object,
                'NewProductName': np.object,
                'pieces':np.float64,
                'weight':np.float64,
                'brand':np.object,
                'Town':np.object,
                'State':np.object
                }


import pickle
def mean_encoding_unique_categorical_features_on_traindata(train):
  #mean_encoding the unique numerical categorical features based on whether the points are outliers are not
  MeanR = train.groupby(['RouteID'], as_index=False)['Demand'].mean()
  MeanSC = train.groupby(['SalesChannelID'], as_index=False)['Demand'].mean()
  MeanSD = train.groupby(['SalesDepotID'], as_index=False)['Demand'].mean()
  MeanP = train.groupby(['ProductID'], as_index=False)['Demand'].mean()
  MeanC = train.groupby(['ClientID'], as_index=False)['Demand'].mean()
  MeanPSD = train.groupby(['ProductID','SalesDepotID'], as_index=False)['Demand'].mean()
  MeanPSC = train.groupby(['ProductID','SalesChannelID'], as_index=False)['Demand'].mean()
  MeanPR = train.groupby(['ProductID','RouteID'], as_index=False)['Demand'].mean()
  MeanPC = train.groupby(['ProductID','ClientID'], as_index=False)['Demand'].mean()
  MeanPCSC = train.groupby(['ProductID','ClientID','SalesChannelID'], as_index=False)['Demand'].mean()
  MeanPCSD = train.groupby(['ProductID','ClientID','SalesDepotID'], as_index=False)['Demand'].mean()
  temp=[MeanR,MeanSC,MeanSD,MeanP,MeanC,MeanPSD,MeanPSC,MeanPR,MeanPC,MeanPCSC,MeanPCSD]

  return temp

import pandas as pd
def merge_newfeatures_of_unique_categorical_data(temp,train_test):
  MeanR=pd.DataFrame(temp[0])
  MeanSC=pd.DataFrame(temp[1])
  MeanSD=pd.DataFrame(temp[2])
  MeanP=pd.DataFrame(temp[3])
  MeanC=pd.DataFrame(temp[4])
  MeanPSD=pd.DataFrame(temp[5])
  MeanPSC=pd.DataFrame(temp[6])
  MeanPR=pd.DataFrame(temp[7])
  MeanPC=pd.DataFrame(temp[8])
  MeanPCSC=pd.DataFrame(temp[9])
  MeanPCSD=pd.DataFrame(temp[10])
  MeanR.rename(columns = {'Demand':'MeanR'}, inplace = True)
  MeanSC.rename(columns = {'Demand':'MeanSC'}, inplace = True)
  MeanSD.rename(columns = {'Demand':'MeanSD'}, inplace = True)
  MeanP.rename(columns = {'Demand':'MeanP'}, inplace = True)
  MeanC.rename(columns = {'Demand':'MeanC'}, inplace = True)
  MeanPSD.rename(columns = {'Demand':'MeanPSD'}, inplace = True)
  MeanPSC.rename(columns = {'Demand':'MeanPSC'}, inplace = True)
  MeanPR.rename(columns = {'Demand':'MeanPR'}, inplace = True)
  MeanPC.rename(columns = {'Demand':'MeanPC'}, inplace = True)
  MeanPCSC.rename(columns = {'Demand':'MeanPCSC'}, inplace = True)
  MeanPCSD.rename(columns = {'Demand':'MeanPCSD'}, inplace = True)
  final_data=pd.merge(train_test,MeanR.astype(object),how = 'left', on='RouteID')
  final_data=pd.merge(final_data,MeanSC.astype(object),how = 'left', on='SalesChannelID')
  final_data=pd.merge(final_data,MeanSD.astype(object),how = 'left', on='SalesDepotID')
  final_data=pd.merge(final_data,MeanP.astype(object),how = 'left', on='ProductID')
  final_data=pd.merge(final_data,MeanC.astype(object),how = 'left', on='ClientID')
  final_data=pd.merge(final_data,MeanPSD.astype(object),how = 'left', on=['ProductID','SalesDepotID'])
  final_data=pd.merge(final_data,MeanPSC.astype(object),how = 'left', on=['ProductID','SalesChannelID'])
  final_data=pd.merge(final_data,MeanPR.astype(object),how = 'left', on=['ProductID','RouteID'])
  final_data=pd.merge(final_data,MeanPC.astype(object),how = 'left', on=['ProductID','ClientID'])
  final_data=pd.merge(final_data,MeanPCSC.astype(object),how = 'left', on=['ProductID','ClientID','SalesChannelID'])
  final_data=pd.merge(final_data,MeanPCSD.astype(object),how = 'left', on=['ProductID','ClientID','SalesDepotID'])
  colname=['ProductID','ClientID','SalesDepotID','SalesChannelID','RouteID']
  final_data.drop(colname, axis=1, inplace=True)# inplace is update test (drop)  
  return final_data

def get_dict_of_ordinal_encoding(value_to_convert):
  key=[]
  value=[]
  for k,v in value_to_convert[0].items():
    key.append(k)
    value.append(v)
  value_to_convert1 = [dict(zip(value , key))]
  return value_to_convert1


import pickle
from sklearn.preprocessing import OrdinalEncoder
def vectorize_categorical_text_features_on_train_data(train):
  leNewClientName=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
  leNewClientName.fit(train['NewClientName'].values.reshape(-1, 1) )
  #print("mapping:",leNewClientName.categories_)
  leNewClientName_map= [dict(enumerate(mapping)) for mapping in leNewClientName.categories_]

  leNewProductName=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
  leNewProductName.fit(train['NewProductName'].values.reshape(-1, 1))
  leNewProductName_map = [dict(enumerate(mapping)) for mapping in leNewProductName.categories_]

  leTown=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
  leTown.fit(train['Town'].values.reshape(-1, 1) )
  leTown_map = [dict(enumerate(mapping)) for mapping in leTown.categories_]

  leState=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
  leState.fit(train['State'].values.reshape(-1, 1))
  leState_map = [dict(enumerate(mapping)) for mapping in leState.categories_]

  lebrand=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
  lebrand.fit(train['brand'].values.reshape(-1, 1))
  lebrand_map = [dict(enumerate(mapping)) for mapping in lebrand.categories_]

  leNewClientName_map1=get_dict_of_ordinal_encoding(leNewClientName_map)
  leNewProductName_map1=get_dict_of_ordinal_encoding(leNewProductName_map)
  leTown_map1=get_dict_of_ordinal_encoding(leTown_map)
  leState_map1=get_dict_of_ordinal_encoding(leState_map)
  lebrand_map1=get_dict_of_ordinal_encoding(lebrand_map)

  temp=[leNewClientName_map1,leNewProductName_map1,leTown_map1,leState_map1,lebrand_map1]

  return temp

def map_categorical_text_features(temp1,train_test):
  leNewClientName_map=temp1[0]
  leNewProductName_map=temp1[1]
  leTown_map=temp1[2]
  leState_map=temp1[3]
  lebrand_map=temp1[4]
  
  train_test['NewClientName']=train_test['NewClientName'].map(leNewClientName_map[0])
  train_test['NewProductName']=train_test['NewProductName'].map(leNewProductName_map[0])
  train_test['Town']=train_test['Town'].map(leTown_map[0])
  train_test['State']=train_test['State'].map(leState_map[0])
  train_test['brand']=train_test['brand'].map(lebrand_map[0])

  return train_test

def preprocess_numerical_data(train_test):
  #the below features are right skewed. Hence took log tranformation to make it as a normal distributed var
  train_test['Demand'] = train_test['Demand'].apply(lambda x:np.log(x + 1))
  train_test['pieces'] = train_test['pieces'].apply(lambda x:np.log(x+1))
  train_test['weight'] = train_test['weight'].apply(lambda x:np.log(x+1))
  return train_test

def preprocess_numerical_data_query(query):
  #the below features are right skewed. Hence took log tranformation to make it as a normal distributed var
  query['pieces']=np.log(int(query['pieces'][0])+1)
  query['weight']=np.log(int(query['weight'][0])+1)
  return query

import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
def getAllTheFeatures(train_test,data):
  train_test=preprocess_numerical_data(train_test)
  temp1=mean_encoding_unique_categorical_features_on_traindata(train_test)
  train_test=merge_newfeatures_of_unique_categorical_data(temp1,train_test)
  temp2=vectorize_categorical_text_features_on_train_data(train_test)
  train_test=map_categorical_text_features(temp2,train_test)
  data=preprocess_numerical_data_query(data)
  data1=merge_newfeatures_of_unique_categorical_data(temp1,data)
  data2=map_categorical_text_features(temp2,data1)
  train_test=train_test.drop(['Demand'],axis=1)
  final_data = data2.append(train_test)
  #print("final_data:",final_data.shape)
  del data,train_test,temp1,temp2,data1,data2
  return final_data

import joblib
def final_f1(query_data):
  train = pd.read_csv('train.csv')
  allFeatures=getAllTheFeatures(train,query_data)
  print(allFeatures.shape)
  print(allFeatures.columns)
  allFeatures=allFeatures.astype(final_types1)
  final_Model= pickle.load(open('final_model.pkl', 'rb'))
  prediction1= final_Model.predict(allFeatures)
  return prediction1
