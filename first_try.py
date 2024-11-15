import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
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
filtered_labels = fmdl.filter_labels(y_all_data, x_all_data)
print(len(train_morgans),len(filtered_labels))
x_train, x_test, y_train, y_test = train_test_split(train_morgans, filtered_labels)

# Optimize
params = {
    'objective': 'binary',        # Clasificación binaria por etiqueta
    'metric': 'binary_logloss',   # Pérdida logarítmica
    'learning_rate': 0.05,        # Tasa de aprendizaje moderada
    'num_leaves': 63,             # Captura interacciones complejas
    'feature_fraction': 0.8,      # Fracción de características para evitar sobreajuste
    'verbose': -1                 # Suprimir logs innecesarios
}

num_boost_round = 300           # Suficientes iteraciones para converger
early_stopping_rounds = 30      # Detener si no mejora en 30 rondas


y_train_np = np.array(y_train)
y_train_y1 = y_train_np[:,0]
y_train_y2 = y_train_np[:,1]
y_train_y3 = y_train_np[:,2]

y_test_np = np.array(y_test)
y_test_y1 = y_test_np[:,0]
y_test_y2 = y_test_np[:,1]
y_test_y3 = y_test_np[:,2]
# This first approach treats the three categories as independent. I know that this is suboptimal, because cross validation should be directed by the F1 score. But let's see how bad this gets
model_y1 = lgb.LGBMClassifier(**params, n_estimators=300)
model_y2 = lgb.LGBMClassifier(**params, n_estimators=300)
model_y3 = lgb.LGBMClassifier(**params, n_estimators=300)

model_y1.fit(x_train, y_train_y1, eval_set=[(x_test, y_test_y1)])
model_y2.fit(x_train, y_train_y2, eval_set=[(x_test, y_test_y2)])
model_y3.fit(x_train, y_train_y3, eval_set=[(x_test, y_test_y3)])

# Predict
y_pred_y1 = model_y1.predict(x_test)
y_pred_y2 = model_y2.predict(x_test)
y_pred_y3 = model_y3.predict(x_test)
# Combine predictions in a single array y1 y2 y3
y_pred = np.column_stack((y_pred_y1, y_pred_y2, y_pred_y3))

# Compute micro F1 score
f1 = f1_score(y_test, y_pred, average='micro')
print(f1)

# Predict for the submission dataset
submission_morgans = fmdl.generate_morgan_fp(x_to_submit, radius, bits)
submission_pred_y1 = model_y1.predict(submission_morgans)
submission_pred_y2 = model_y2.predict(submission_morgans)
submission_pred_y3 = model_y3.predict(submission_morgans)
submission_pred = np.column_stack((submission_pred_y1, submission_pred_y2, submission_pred_y3))
# Save it in a Df with 2 columns. Make sure that the indices of the DF are the same as those we read from the submission dataset
submission_df = pd.DataFrame(submission_pred, index=x_to_submit.index, columns=y_all_data.columns)
submission_df.to_csv('submission.csv')
