import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

df = pd.read_csv('ai4i2020.csv')
df = df.drop(['UDI', 'Product ID'], axis=1)

df.columns = [col.replace('[', '').replace(']', '').replace(' ', '_').replace('<', '') for col in df.columns]

le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

df['Temp_Diff'] = df['Process_temperature_K'] - df['Air_temperature_K']

df['Power'] = df['Torque_Nm'] * df['Rotational_speed_rpm']

df['Strain'] = df['Torque_Nm'] * df['Tool_wear_min']

features_to_drop = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df_clean = df.drop(features_to_drop, axis=1)

X = df_clean.drop('Machine_failure', axis=1)
y = df_clean['Machine_failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

counter = float(y_train.value_counts()[0] / y_train.value_counts()[1])

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=counter,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Wyniki Zoptymalizowanego Modelu XGBoost")
print(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Brak awarii', 'Awaria'])
disp.plot(cmap='Greens', ax=ax)
plt.title('Macierz pomyłek - Model XGBoost')
plt.show()

plt.figure(figsize=(10, 6))
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(6).sort_values().plot(kind='barh', color='darkgreen')
plt.title('Najważniejsze czynniki awarii (XGBoost)')
plt.xlabel('Waga cechy')
plt.tight_layout()
plt.show()