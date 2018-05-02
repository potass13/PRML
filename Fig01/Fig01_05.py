# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rms(p, df):
	error = 0.
	for index, list in df.iterrows():
		error += 0.5*(np.polyval(p, list.x)-list.t)**2
	return np.sqrt(2.*error/len(df))

def main():
	M = range(0, 10)
	N = len(M)
	sigma = 0.2
	
	xx1 = np.linspace(0, 1, N)
	tt1 = np.sin(2.*np.pi*xx1)+np.random.normal(0, sigma, len(xx1))
	df_train = pd.DataFrame(np.array([xx1, tt1]).transpose(), columns=['x', 't'])
	
	xx2 = np.linspace(0, 1, 100)
	tt2 = np.sin(2.*np.pi*xx2)+np.random.normal(0, sigma, len(xx2))
	df_test = pd.DataFrame(np.array([xx2, tt2]).transpose(), columns=['x', 't'])
	
	rms_train = []
	rms_test = []
	
	for m in M:
		p_train = np.polyfit(df_train.x.values, df_train.t.values, m)
		rms_train.append(rms(p_train, df_train))
		rms_test.append(rms(p_train, df_test))
	
	plt.xlabel('M')
	plt.ylabel('E_rms')
	plt.xlim(-0.1, 9.1)
	plt.ylim(0, 1.)
	
	plt.plot(M, rms_train, color='blue', marker='o', label='training set')
	plt.plot(M, rms_test, color='red', marker='o', label='test set')
	plt.legend(loc='upper left', fontsize=15, numpoints=1)
	
	plt.show()

if __name__ == '__main__':
	main()
