"""Personality Type Classifier: An ML Pipeline to Predict Introvert vs Extrovert Traits"""

# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import joblib


# Loading Dataset
df = pd.read_csv('personality_dataset.csv')

# Doing Label Encoding Manually
df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert':0})

# Detecting Missing Values
df.isna().mean()*100            # After printing, found that Each column contain < 3% missing values

# Dropping Missing Values
df.dropna(inplace = True)

# Detecting Duplicate rows
df.duplicated().sum()            # After printing, found 379 Duplicated rows

# Dropping Duplicate rows
df = df.drop_duplicates()

# Separating Data into features (X) and target (y)
X =  df.drop(columns=['Personality'])
y = df['Personality']

# Splitting features and target in train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 42)

# Selecting Numerical and Categorical Columns
cat = X.select_dtypes(exclude = 'number').columns
num = X.select_dtypes(include = 'number').columns

# Creatng Categorical Pipeline to transform Categorical Columns
cat_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown = 'ignore'))
])

# Creatng Numerical Pipeline to transform numerical Columns

num_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Columntransformer
preprocessor = ColumnTransformer(transformers=[
    ('cat', cat_pipeline , cat),
    ('num', num_pipeline, num)
])

# Final Pipeline
final_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression())])

# Creating Param_grid
param_grid = {
    'model__C': [0.1, 1, 10],
    'model__solver': ['lbfgs', 'saga']
}

# Determining Best Estimator
grid = GridSearchCV(final_pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
model = grid.fit(X_train, y_train).best_estimator_

# Making Prediction on Test Data
y_pred = model.predict(X_test)

# Classifying whether Model is underfitting or overfitting
print(f'Training Score: {model.score(X_train, y_train)}')
print(f'Testing Score: {model.score(X_test, y_test)}')

# EValuation Metrics
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)}')
print(f'Precision Score: {precision_score(y_test, y_pred)}')
print(f'Recall Score: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

# Saving Model
joblib.dump(model, 'personality_classifier_model.pkl')