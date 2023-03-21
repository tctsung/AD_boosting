import pandas as pd
import numpy as np



def vote_hyperparam(cv_results_, top_n=5):
	"""
	TODO: select optimal hyperparameter from GridSearchCV result by ensemble learning
	cv_results_: .cv_results_ attribute from GridSearchCV() object
	"""
	assert top_n % 2 == 1, "Number of votes shouldn't be even number"
	if not isinstance(cv_results_, pd.DataFrame): cv_results_ = pd.DataFrame(cv_results_)
	df = pd.DataFrame(cv_results_).sort_values("rank_test_score")
	df = df.iloc[:top_n].loc[:,df.columns.str.contains("param_")]  # keep top n rows
	output = df.mode()   # get mode
	output.columns = output.columns.str.replace("param_","")  # to match param nm for dict
	output = output.to_dict('records')[0]  # turn to dict
	return output
