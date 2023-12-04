import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
def sort_dataset(dataset_df):
	#TODO: Implement this function
	return dataset_df.sort_values(by=['year'])
def split_dataset(dataset_df):	
	#TODO: Implement this function
	X=dataset_df.drop(columns='salary',axis=1)
	Y=dataset_df['salary']
	X_train=X.iloc[:1718,:]
	X_test=X.iloc[1718:,:]
	Y_train=Y.iloc[:1718]*0.001
	Y_test=Y.iloc[1718:]*0.001
	return X_train,X_test,Y_train,Y_test
def extract_numerical_cols(dataset_df):
	#TODO: Implement this function
	return dataset_df[['age','G','PA','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','HBP','SO','GDP','fly','war']]
def train_predict_decision_tree(X_train, Y_train, X_test):
	#TODO: Implement this function
	dt_r=DecisionTreeRegressor()
	dt_r.fit(X_train,Y_train)
	return dt_r.predict(X_test)
def train_predict_random_forest(X_train, Y_train, X_test):
	#TODO: Implement this function
	rf_r=RandomForestRegressor()
	rf_r.fit(X_train,Y_train)
	return rf_r.predict(X_test)
def train_predict_svm(X_train, Y_train, X_test):
	#TODO: Implement this function
	svm_pipe=make_pipeline(
		StandardScaler(),
		SVR()
	)
	svm_pipe.fit(X_train,Y_train)
	return svm_pipe.predict(X_test)
def calculate_RMSE(labels, predictions):
	#TODO: Implement this function
	return np.sqrt(np.mean((labels-predictions)**2))
if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))