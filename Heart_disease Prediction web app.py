# -*- coding: utf-8 -*-
"""
Created on Sat May 25 18:03:58 2024

@author: Lenovo
"""

import numpy as np
import pickle
import streamlit as st

# load saved model
loaded_model = pickle.load(open('C:/Users\Lenovo/Data Modeling/UAS/Random_Forest.sav', 'rb'))


# creating function or prediction
def Heart_disease_prediction(coba):
    
    cobaRF = loaded_model.predict(coba)
    print("Dengan menggunakan model Algorithm Random Forest")
    for prediction in cobaRF:
        if prediction == 0:
            return "Diprediksi pasien tidak menderita serangan jantung"
        else:
            return "Diprediksi pasien menderita serangan jantung"

def main():

    #give title
    st.title('Heart disease Prediction web app')
    
    #getting input data from user
    
    age = st.text_input('Age of the patient')
    sex = st.text_input('sex of the patient [1: Male, 0: Female]')
    cp = st.text_input('Chest pain type 4 values [0: No chest pain, 4: Chest pain]')    
    trestbps = st.text_input(' Resting blood pressure [mm Hg]')
    chol = st.text_input('Serum cholesterol [mg/dl]')
    fbs = st.text_input('Fasting blood sugar [1: true, 0: false]')
    restecg = st.text_input('Esting electrocardiogram results [values 0,1,2]')
    thalach = st.text_input('Maximum heart rate achieved [Numeric value between 60 and 202]')
    exang = st.text_input('Exercise induced angina [1: yes, 0: no]')
    oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    
    #code for prediction
    diagnosis = ' '
    
    #creating button for prediction
    
    if st.button('Heart Disease Result'):
        diagnosis = Heart_disease_prediction(np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]).reshape(1, -1))
        
    st.success(diagnosis)


if __name__ == '__main__':
    main()        
