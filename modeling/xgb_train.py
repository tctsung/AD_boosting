import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
import pickle
def load_py(location):
	file = open(f'{location}', 'rb')     # open a file, where you stored the pickled data
	data = pickle.load(file)
	file.close()
	return data
def save_py(obj, location):
	"""
	TODO: to save a python object
	:param obj: object name
	:param location: location to save the file
	"""
	if location[-4:] != '.pkl': location += '.pkl'  # add file extension
	savefile = open(f'{location}', 'wb')
	pickle.dump(obj, savefile)
	savefile.close()
def xgboost_greedy(data, label, params, fold=5, seed=111):
    model = xgb.XGBClassifier(objective='binary:logistic',
                              random_state=seed)
    model_greedy = GridSearchCV(estimator = model, param_grid=params,cv=fold, 
                                verbose=500, scoring = 'matthews_corrcoef'  # outcome imbalanced
                                )
    model_greedy.fit(data, label) 
    output = pd.DataFrame(model_greedy.cv_results_).sort_values("rank_test_score")
    print(output.head(10))
    return output
def main():  
	# inputs:
	# data = load_py('../../clean_data/train1.pkl')
	data = load_py("train1_dummy.pkl")
	x = data.copy()
	y = x.pop('progress')
	# hyperparams:
	params = {
	    'n_estimators':[100,300,1000],
	    'reg_lambda':[0,0.1,0.5,1,2,3],       # L2 regularization               
	    'learning_rate':[0.01, 0.05, 0.1, 0.5, 1], 
	    'max_depth':list(range(3,10,2)),             # tree depth 
	    'tree_method':["exact", "hist"],
	    'colsample_bytree':[0.7,1],
	    'subsample':[0.7,1]
	}
	# training:
	output = xgboost_greedy(x, y, params)
	save_py(output, "xgb_grid_result")

if __name__ == '__main__':
	main()
