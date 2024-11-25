# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# load saved model
loaded_model = pickle.load(open('C:/Users\Lenovo/Data Modeling/UAS/Random_Forest.sav', 'rb'))

coba = [[0.479167,	1.0,	0.000000,	0.292453,	0.196347,	0.0,	0.5,	0.740458,	0.0,	0.1612]]

cobaRF = loaded_model.predict(coba)
print("Dengan menggunakan model Algorithm Random Forest")
for prediction in cobaRF:
    if prediction == 0:
        print("Diprediksi pasien tidak menderita serangan jantung")
    else:
        print("Diprediksi pasien menderita serangan jantung")