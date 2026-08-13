"""
RESEARCH / PROTOTYPE CODE - Before Refactoring
This is the exploratory, notebook-style code written during the research phase.
It works, but lacks proper structure, error handling, logging, and separation of concerns.
Contrast this with the production code in src/.
"""

# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# %%  Load data - no validation, no logging
iris = load_iris()
X = iris.data
y = iris.target
print("Data shape:", X.shape)
print("Classes:", iris.target_names)

# %% Quick EDA
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.describe())
print(df['target'].value_counts())

# %% Split - hardcoded params, no stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Scale - manual, not in pipeline
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% Train model - magic numbers, no logging
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# %% Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% Save - no error handling
joblib.dump({'clf': clf, 'scaler': scaler}, 'research_code/model.pkl')
print("Saved!")

# %% Predict new sample - raw numpy, no schema
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
pred = clf.predict(sample_scaled)
print("Prediction:", iris.target_names[pred[0]])
