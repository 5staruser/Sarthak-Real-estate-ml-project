import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import dump,load
from sklearn.metrics import mean_squared_error
housing=pd.read_csv("housing.csv")
# Train-Test split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["CHAS"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
housing=strat_train_set.copy()
housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)
X=imputer.transform(housing)
housing_tr=pd.DataFrame(X,columns=housing.columns)
# Creating a pipeline
my_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("stdscaler",StandardScaler()),
])
housing_num_tr=my_pipeline.fit_transform(housing)
# Selecting a model
models=input("Select the type of model : ")
if models=="Linear Regression":
    model=LinearRegression()
elif models=="Decision Tree Regressor":
    model=DecisionTreeRegressor()
else:
    model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)
some_labels=housing_labels.iloc[:5]
some_data=housing.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
# Evaluating the model
housing_prediction=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_prediction)
rmse=np.sqrt(mse)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
def printscores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard Deviation: ",scores.std())
printscores(rmse_scores)
# Saving the model
dump(model,"Sarthak.joblib")
# Testing model on Test data
x_test=strat_test_set.drop("MEDV",axis=1)
y_test=strat_test_set["MEDV"].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_prediction=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_prediction)
final_rmse=np.sqrt(final_mse)
print(final_prediction,list(y_test))