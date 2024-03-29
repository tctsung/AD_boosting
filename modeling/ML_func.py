import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# decorator to silence warning:
# deprecated warning in light GBM
import warnings
from functools import wraps
def silence_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            result = func(*args, **kwargs)
        return result
    return wrapper

def vote_hyperparam(cv_results_, top_n=5):
	"""
	TODO: select optimal hyperparameter from GridSearchCV result by ensemble learning
	cv_results_: .cv_results_ attribute from GridSearchCV() object
	top_n: top n rank will be used to select optimal params
	"""
	if not isinstance(cv_results_, pd.DataFrame): cv_results_ = pd.DataFrame(cv_results_)
	df = pd.DataFrame(cv_results_).sort_values("rank_test_score")
	df = df.loc[df['rank_test_score']<=top_n ,df.columns.str.contains("param_")]  # keep top n rows
	output = df.mode()   # get mode
	output.columns = output.columns.str.replace("param_","")  # to match param nm for dict
	output = output.to_dict('records')[0]  # turn to dict
	return output

def plt_learning_curve(train_loss, test_loss):
	"""
	TODO: plot learning curves
	:param train_loss, test_loss: list
	"""
	fig, ax = plt.subplots()
	ax.plot(train_loss, label='Train Error')
	ax.plot(test_loss, label='Validation Error')
	ax.set_title('Learning Curve')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss')
	min_val_loss_idx = test_loss.index(min(test_loss))  # add veritical line with lowest loss
	ax.axvline(x=min_val_loss_idx, color='r', linestyle='--')
	ax.legend()
	plt.show()
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
import catboost as cat
def catboost_greedy(data, label, params, cat_features, fold=5, seed=111):
	model = cat.CatBoostClassifier(cat_features=cat_features, random_seed=seed, 
									 verbose=0, early_stopping_rounds=100)
	model_greedy = GridSearchCV(estimator = model, cv=fold, param_grid=params, 
								scoring = 'matthews_corrcoef' # outcome imbalanced
								)
	model_greedy.fit(data, label)
	output = pd.DataFrame(model_greedy.cv_results_).sort_values("rank_test_score")
	display(output.head(10))
	return output
import lightgbm as lgb
def lightgbm_greedy(data, label, params, fold=5, seed=111):


	model = lgb.LGBMClassifier(objective='binary', seed=seed)
	model_greedy = GridSearchCV(estimator = model, param_grid=params,cv=fold, 
								verbose=500, scoring = 'matthews_corrcoef'  # outcome imbalanced
								)
	model_greedy.fit(data, label) 
	output = pd.DataFrame(model_greedy.cv_results_).sort_values("rank_test_score")
	display(output.head(10))
	return output
import xgboost as xgb
def xgboost_greedy(data, label, params, fold=5, seed=111):
    model = xgb.XGBClassifier(objective='binary:logistic',
                              random_state=seed)
    model_greedy = GridSearchCV(estimator = model, param_grid=params,cv=fold, 
                                verbose=500, scoring = 'matthews_corrcoef'  # outcome imbalanced
                                )
    model_greedy.fit(data, label) 
    output = pd.DataFrame(model_greedy.cv_results_).sort_values("rank_test_score")
    display(output.head(10))
    return output
from sklearn.model_selection import StratifiedKFold
def catboost_cv(data, label, params, cat_features, fold=5, seed=555, plotit=True):
	"""
	TODO: CV for input data and return the best CatBoostClassifier model
	"""
	models = {}
	skf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
	i = 0   # save model label
	lowest_loss = float('inf')
	if "loss_function" not in params: 
		params["loss_function"] = 'Logloss'
	loss_func = params["loss_function"]
	for train_index, test_index in skf.split(data, label):
		x_train = data.loc[train_index,:]
		y_train = label[train_index]
		x_test = data.loc[test_index,:]
		y_test = label[test_index]
		model = cat.CatBoostClassifier(cat_features=cat_features, 
									   random_seed=seed+i+100, **params)
		model.fit(x_train, y_train, verbose=0, use_best_model=True, 
			eval_set=[(x_train, y_train), (x_test, y_test)])
		validate_loss = model.evals_result_['validation_1'][loss_func]
		curr_loss = np.min(validate_loss)  # lowest loss in current model
		models[str(i)] = model # save model
		if lowest_loss > curr_loss:
			lowest_loss = curr_loss   # update lowest loss
			best_id = i
		i += 1
	best_model = models[str(best_id)]
	if plotit:
		train_loss = best_model.evals_result_['validation_0'][loss_func]
		test_loss = best_model.evals_result_['validation_1'][loss_func]
		plt_learning_curve(train_loss, test_loss)
	print(f"Setup: {params}, {fold}-fold CV with stratification")
	return best_model
@silence_warnings
def lightgbm_cv(data, label, params, cat_features, fold=5, seed=555, plotit=True):
	"""
	TODO: CV for input data and return the best CatBoostClassifier model
	"""
	if cat_features:            # change dtypes for categorical features
		for c in cat_features:
			data[c] = data[c].astype('category')
	models = {}
	skf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
	i = 0   # save model label
	lowest_loss = float('inf')
	if not 'early_stopping_round' in params:  # must set early stopping to predict with best model
		if 'n_estimators' in params:
			params['early_stopping_round'] = (params['n_estimators']//10)
		else:
			params['early_stopping_round'] = 100
	for train_index, test_index in skf.split(data, label):
		x_train = data.loc[train_index,:]
		y_train = label[train_index]
		x_test = data.loc[test_index,:]
		y_test = label[test_index]
		model = lgb.LGBMClassifier(objective='binary', random_state=seed+i+100, **params)
		model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
				  callbacks=[lgb.callback.log_evaluation(period=0, show_stdv=False)]) # silence model
		validate_loss = model.evals_result_['valid_1']['binary_logloss']
		curr_loss = validate_loss[(model.best_iteration_-1)]    # lowest loss in current model
		models[str(i)] = model        # save model
		if lowest_loss > curr_loss:
			lowest_loss = curr_loss   # update lowest loss
			best_id = i
		i += 1
	best_model = models[str(best_id)]
	if plotit:
		train_loss = best_model.evals_result_['training']['binary_logloss']
		validate_loss = best_model.evals_result_['valid_1']['binary_logloss']
		plt_learning_curve(train_loss, validate_loss)
	print(f"Setup: {params}, {fold}-fold CV with stratification")
	return best_model
def xgboost_cv(data, label, params, fold=5, seed=555, plotit=True):
	"""
	TODO: CV for input data and return the best CatBoostClassifier model
	"""
	models = {}
	skf = StratifiedKFold(n_splits=fold, random_state=seed, shuffle=True)
	i = 0   # save model label
	lowest_loss = float('inf')
	if not 'early_stopping_rounds' in params:  # must set early stopping to predict with best model
		if 'n_estimators' in params:
			params['early_stopping_rounds'] = (params['n_estimators']//10)
		else:
			params['early_stopping_rounds'] = 100
	for train_index, test_index in skf.split(data, label):
		x_train = data.loc[train_index,:]
		y_train = label[train_index]
		x_test = data.loc[test_index,:]
		y_test = label[test_index]
		model = xgb.XGBClassifier(objective='binary:logistic', eval_metric = 'logloss',
								  random_state=seed+i+100, **params)
		model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
				  verbose=0) # silence model
		validate_loss = model.evals_result()['validation_1']['logloss']
		curr_loss = validate_loss[(model.best_iteration-1)]    # lowest loss in current model
		models[str(i)] = model        # save model
		if lowest_loss > curr_loss:
			lowest_loss = curr_loss   # update lowest loss
			best_id = i
		i += 1
	best_model = models[str(best_id)]
	if plotit:
		train_loss = best_model.evals_result()['validation_0']['logloss']
		validate_loss = best_model.evals_result()['validation_1']['logloss']
		plt_learning_curve(train_loss, validate_loss)
	print(f"Setup: {params}, {fold}-fold CV with stratification")
	return best_model
















