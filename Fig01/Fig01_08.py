# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def y(w, x):
	y = 0.
	for i, ww in enumerate(w):
		y += ww*(x**i)
	return y

def rms_error(p, df, f):
	error = 0.
	for index, list in df.iterrows():
		error += 0.5*(f(p, list.x)-list.t)**2
	return np.sqrt(2.*error/len(df))

def main():
	M = 9
	N = 10
	lnL = np.linspace(-37, -20)
	Lbd = np.exp(lnL)
	sigma = 0.3
	
	xx1 = np.linspace(0, 1, N)
	tt1 = np.sin(2.*np.pi*xx1)+np.random.normal(0, sigma, len(xx1))
	df_train = pd.DataFrame(np.array([xx1, tt1]).T, columns=['x', 't'])
	
	xx2 = np.linspace(0, 1, N)
	tt2 = np.sin(2.*np.pi*xx2)+np.random.normal(0, sigma, len(xx2))
	df_test = pd.DataFrame(np.array([xx2, tt2]).T, columns=['x', 't'])
	
	rms_train = []
	rms_test = []
	
	for i, lbd in enumerate(Lbd):
		Phi = np.array([df_train.x**k for k in range(0, M+1)]).T
		w = np.dot(np.dot(np.linalg.inv(lbd*np.eye(M+1)+np.dot(Phi.T, Phi)), Phi.T), df_train.t)
		rms_train.append(rms_error(w, df_train, y))
		rms_test.append(rms_error(w, df_test, y))
	
	plt.xlabel('ln $\lambda$')
	plt.ylabel('Erms')
	plt.xlim(np.min(lnL), np.max(lnL))
	plt.ylim(0, 1.0)
	
	plt.plot(lnL, rms_train, color='blue', label='training set')
	plt.plot(lnL, rms_test, color='red', label='test set')
	plt.legend(loc='upper left', fontsize=15, numpoints=1)
	plt.show()

if __name__ == '__main__':
	main()
