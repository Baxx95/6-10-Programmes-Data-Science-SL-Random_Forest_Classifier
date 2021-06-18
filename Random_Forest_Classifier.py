# -*- coding: utf-8 -*-
"""
Created on Wed May 12 04:34:12 2021

@author: Zakaria
"""

import pandas as pd

data = pd.read_csv('prediction_de_fraud_2.csv')

caracteristiques = data.drop('isFraud', axis=1).values

cible = data['isFraud'].values

from sklearn.preprocessing import LabelEncoder

LabEncdr_X = LabelEncoder()
caracteristiques[:, 1] = LabEncdr_X.fit_transform(caracteristiques[:, 1])
caracteristiques[:, 3] = LabEncdr_X.fit_transform(caracteristiques[:, 3])
caracteristiques[:, 6] = LabEncdr_X.fit_transform(caracteristiques[:, 6])


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(caracteristiques, cible, test_size=.3, random_state=50)


from sklearn.ensemble import RandomForestClassifier

Random_frst_cls = RandomForestClassifier(random_state=50)

Random_frst_cls.fit(x_train, y_train)

Random_frst_cls.score(x_test, y_test) ## ==> 0.9550561797752809

