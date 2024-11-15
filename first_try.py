import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import fingerprint_module as fmdl

# Load dataset of X's
x_all_data = pd.read_csv('dataset/X_train_UbsWnSC.csv',index_col=0)
y_all_data = pd.read_csv('dataset/y_train_2SdpCfw.csv',index_col=0)
x_to_submit = pd.read_csv('dataset/X_test_W8QYD44.csv',index_col=0)

# print(x_all_data)

# We compute the molecular fingerprints. A module contains different fingerprint creation functions
radius = 2
bits = 2048
train_morgans = fmdl.generate_morgan_fp(x_all_data, radius, bits)

# Split the dataset in train and test
x_train, x_test, y_train, y_test = train_test_split(train_morgans,)
