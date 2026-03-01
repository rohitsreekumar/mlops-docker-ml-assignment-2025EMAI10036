import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
data = pd.read_csv(url)

# Drop missing values
data = data.dropna()

# Encode categorical target
le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

# Features and target
X = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = data['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Precision
precision = precision_score(y_test, y_pred, average='macro')

# AUC score
auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print("Model Precision:", precision)
print("Model AUC Score:", auc)
