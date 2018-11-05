import pandas as pd
import ast


class ReadNestedList:

	def __init__(self, tweets, target, name):
		self.tweets = tweets
		self.target = target
		self.name = name
		self.all_elements = list()
		self.df = None
		self.grpDF = None

	def read(self):
		for el in self.target:
			el = ast.literal_eval(el)
			if len(el) > 0:
				self.all_elements.extend(el)
		return self

	def DF(self):
		self.df = pd.DataFrame(self.all_elements, columns=[self.name])
		self.df['{}_lower_case'.format(self.name)] = self.df[self.name].str.lower()
		return self

	def computeGrpDF(self):
		self.grpDF = self.df.groupby('{}_lower_case'.format(self.name))['{}_lower_case'.format(self.name)].count().reset_index(name = 'count').sort_values(['count'], ascending=False)
		return self
