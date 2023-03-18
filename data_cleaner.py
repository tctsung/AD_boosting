import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
class Cleaner:
	"""
	For data cleaning
	"""
	def __init__(self, df):
		assert isinstance(df, pd.DataFrame), "Input data must be a pandas DataFrame."
		self.df = df
	# def __rp
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



# def auto_dtype(df):
# 	"""
	
# 	"""
# 	return df.apply(lambda x:x.dtype)



