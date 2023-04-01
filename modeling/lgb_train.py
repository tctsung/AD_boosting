import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
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
def lightgbm_greedy(data, label, params, fold=5, seed=111):
	model = lgb.LGBMClassifier(objective='binary', seed=seed)
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
	data = load_py("train1.pkl")
	x = data.copy()
	y = x.pop('progress')
	cat_features = ['PTGENDER', 'PTETHCAT', 'PTMARRY', 'PTRACCAT']
	for c in cat_features:
		x[c] = x[c].astype('category')
	# hyperparams:
	params = {
    'num_leaves':[3,7,15,31],           # key to ctrl overfit; theoretically=2^(max_depth)
    # 'max_depth': [-1]                 # no need to tune because leaf-wise growth
    'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1],       # default 0.1
    'n_estimators':[100, 150, 300, 500],  # not sure if return best model 
    'colsample_bytree':[0.7,1],
    'subsample':[0.7,1],
    'is_unbalance':[False, True],     # auto-weight for imbalance classes
    'reg_lambda':list(map(lambda x:x/2, range(0,11)))   # L2 regularization
	}
	# training:
	output = lightgbm_greedy(x, y, params)
	save_py(output, "lgb_grid_result")

if __name__ == '__main__':
	main()
