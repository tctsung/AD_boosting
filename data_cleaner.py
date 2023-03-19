import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import pickle
class Cleaner:
	"""
	For data cleaning
	"""
	def __init__(self, df):
		assert isinstance(df, pd.DataFrame), "Input data must be a pandas DataFrame."
		self.df = df
	# def __repr__
	def rm_na_cols(self, na_percentage):
		assert 0 <= na_percentage <= 1, "na_percentage must be between 0 and 1"
		cnt = int(self.df.shape[0] * na_percentage)
		self.df.dropna(thresh = cnt, axis=1, inplace=True)
	def rm_no_outcome(self, nm=None):
		"""
		TODO: rm rows that don't have outcome value
		:param nm: name of outcome feature in dataset
		"""
		if nm:
			self.df.dropna(subset=[nm], inplace=True)
	def binary_to_int(self):
		"""
		TODO: turn binary variable to integer
		"""
		int_col = self.df.columns[self.df.apply(lambda x:len(x.dropna().unique()))==2].to_list() # find unique val=2 not including NaN
		self.df.loc[:,int_col] = self.df.loc[:,int_col].astype('Int64')
	def fix_colname(self):
		"""
		TODO: In column names, turn uncommon symbols to underline. Spaces will be replaced by underline
		"""
		new_nms = list(map(lambda col: re.sub(r'[^0-9a-zA-Z_]', '_', col).lower(), self.df.columns))
		self.df.columns = new_nms
		return self.df
	def auto_clean(self, na_percentage=0.99, nm=None):
		self.drop_duplicates(inplace=True) # rm duplicate rows
		self.rm_no_outcome(nm)             # drop rows with no outcome
		self.rm_na_cols(na_percentage)     # drop cols with missingness > na_percentage
		self.binary_to_int()               # turn binary variables to integer
		self.fix_colname()                 # rm weird symbols from column names
def overview(df):
	"""
	Check the data types & NA percentage & unique values of the data
	"""
	print(f'size: {df.shape}')
	output = df.apply(lambda x: (x.dtype,x.isna().mean(),x.unique()), axis=0).T
	output.columns = ["dtype", "NaN_percentage","unique"]
	return output.sort_values(by="NaN_percentage", ascending=False)
def standardization(df):
	"""
	TODO: standardized the numeric variables in the data; dtype!=float won't be transformed
	"""
	raw_mean_sd = (df.describe()).iloc[1:3,:]  # mean & sd of raw data
	float_col = df.select_dtypes(include=['float64', 'float32'])
	scaler = StandardScaler()
	scaler.fit(float_col)
	df.loc[:, float_col.columns] = scaler.transform(float_col)   # replace by scaled data
	return df, raw_mean_sd




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
def load_py(location):
	file = open(f'{location}', 'rb')     # open a file, where you stored the pickled data
	data = pickle.load(file)
	file.close()
	return data







