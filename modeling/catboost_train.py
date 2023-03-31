import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import catboost as cat
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
def catboost_greedy(data, label, params, cat_features, fold=5, seed=111):
	model = cat.CatBoostClassifier(cat_features=cat_features, random_seed=seed, 
									 early_stopping_rounds=100)
	model_greedy = GridSearchCV(estimator = model, cv=fold, param_grid=params, verbose=100, 
								scoring = 'matthews_corrcoef' # outcome imbalanced
								)
	model_greedy.fit(data, label)
	output = pd.DataFrame(model_greedy.cv_results_).sort_values("rank_test_score")
	return output
def main():   # code to run
	# data = load_py("train1.pkl")
	data = load_py('../../clean_data/train1.pkl')
	x = data.copy()
	y = x.pop('progress')
	# inputs:
	# params = {
	#     'depth':list(range(3,11)),                       # 6-10 is recommend
	#     'learning_rate':[0.003,0.006, 0.009,0.03,0.06,0.09,0.3],   # default: 0.03
	#     'iterations':[100,200,300,500,1000,2000],     # because couldn't apply early stopping
	#     'l2_leaf_reg':list(map(lambda x:x/2, range(0,13))),  # default 3
	# }
	params = {
		'depth':list(range(3,11))
	}
	cat_features = ['PTGENDER', 'PTETHCAT', 'PTMARRY', 'PTRACCAT']
	# training:
	output = catboost_greedy(x, y, params, cat_features)
	save_py(output, "catboost_grid_result_test")

if __name__ == '__main__':
	main()
