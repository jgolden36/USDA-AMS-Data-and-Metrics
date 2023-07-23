import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
os.chdir('C:\\Users\\bnhas\\OneDrive\\Desktop\\Classes\\Work Summer 2023\\AMS Comp\\Data')
df1=pd.read_csv('food_access1.csv')
df2=pd.read_csv('food_access2.csv')
df3 = pd.concat([df1, df2], axis = 0)
df3=df3.drop_duplicates(subset=['county_name','state_name','variable_name'])
df3=df3.pivot(index=['county_name','state_name'], columns='variable_name', values='value').reset_index()
#df3=pd.melt(df3, id_vars=[['county_name','state_name']],value_vars=[['value']])
#pct_laccess_pop
X=df3.drop(['county_name','state_name'], axis=1)
X=X[['work_inside_state_out_of_county','work_outside_state','disability_total','High_food_Insecurity']]
X=X.dropna()
y=X['High_food_Insecurity'].to_numpy()
X=X.drop(['High_food_Insecurity'], axis=1).to_numpy()
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X, y)
prediction=clf.predict(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
df = pd.concat([principalDf, pd.DataFrame(y, columns = ['High_food_Insecurity'])], axis = 1)
df = pd.concat([df, pd.DataFrame(prediction, columns = ['Prediction'])], axis = 1)
fig = px.scatter(df, x="principal component 1", y="principal component 2", color="High_food_Insecurity", symbol="Prediction")
fig.show()