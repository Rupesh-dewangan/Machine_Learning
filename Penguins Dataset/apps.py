import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🐧Penguine Species Prediction')
st.info('Predicting the penguine species using Machine Learning model')

with st.expander('Data'):
    st.write("**Raw data**")
    df = pd.read_csv('data.csv')
    df

    st.write('**X**')
    X_raw = df.drop('species',axis=1)
    X_raw

    st.write('**y**')
    y_raw = df.species
    y_raw


with st.expander('Data Visualization'):
    st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

# data preparation

with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island', ('Biscoe','Dream','Torgersen'))
    sex = st.selectbox('Gender',('MALE','FEMALE'))
    bill_length_mm = st.slider('Bill Length (mm)',32.1,59.6,43.9)

    bill_depth_mm = st.slider('Bill Depth (mm)',13.1,21.5,17.2)
    flipper_length_mm = st.slider('Flipper Length (mm)',172.0,231.0,201.0)
    body_mass_g = st.slider('Body Mass (g)',2700.0,6300.0,4207.0)

    # Create a dataframe for the input feature

    data = {'island': island,
            'bill_length_mm':bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm' :flipper_length_mm,
            'body_mass_g':body_mass_g,
            'sex' :sex

            
            }
input_df = pd.DataFrame(data,index=[0])
input_penguins = pd.concat([input_df,X_raw],axis=0)


with st.expander('Input features'):
    st.write('**Input Penguins**')
    input_df
    st.write('**Combine Penguins Data**')
    input_penguins

#encode X 
encode = ['island','sex']
df_penguins = pd.get_dummies(input_penguins,prefix=encode)

X = df_penguins[1:]

input_row = df_penguins[:1  ]

# Encode Y 
target_mapper = {'Adelie':0, 'Chinstrap':1 ,'Gentoo':2}

def target_encode(val):
    return target_mapper[val]

y= y_raw.apply(target_encode)


with st.expander('Data Preparation'):
    st.write('**Encoded input penguins (x)**')
    input_row  
    st.write('**Encoded y**')
    y

#Model Training and inference
## Train the ML Model

clf = RandomForestClassifier()
clf.fit(X,y)

## Apply model 

prediction = clf.predict(input_row)
prediction_prob = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_prob)
df_prediction_proba.columns = ['Adelie', 'Chinstrap','Gentoo']
#df_prediction_proba.rename(columns={0:'Adelie', 1:'Chinstrap' ,2:'Gentoo'})



# Display predicted species 
st.subheader('Predicted Species')
st.dataframe(df_prediction_proba, column_config={
    'Adelie': st.column_config.ProgressColumn('Adelie',format='%f',width='medium',min_value=0,max_value=1),
    'Chinstrap': st.column_config.ProgressColumn('Chinstrap',format='%f',width='medium',min_value=0,max_value=1),
    'Gentoo': st.column_config.ProgressColumn('Gentoo',format='%f',width='medium',min_value=0,max_value=1)
})

penguins_species = np.array(['Adelie', 'Chinstrap','Gentoo'])
st.success(str(penguins_species[prediction][0]))