# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def y(w, x):
	y = 0.
	for i, ww in enumerate(w):
		y += ww*(x**i)
	return y

def main():
	M = 9
	N = 10
	lnL = [np.nan, -18., 0.]
	Lbd = np.exp(lnL)
	Lbd[np.isnan(Lbd)] = 0
	sigma = 0.2
	
	x = np.linspace(0, 1, N)
	t = np.sin(2.*np.pi*x)+np.random.normal(0, sigma, len(x))
	
	xx = np.linspace(0, 1, 100)
	
	for i, lbd in enumerate(Lbd):
		Phi = np.array([x**k for k in range(0, M+1)]).T
		w = np.dot(np.dot(np.linalg.inv(lbd*np.eye(M+1)+np.dot(Phi.T, Phi)), Phi.T), t)
		
		plt.subplot(1, 3, i+1)
		plt.xlabel('x')
		plt.ylabel('t')
		plt.xlim(-0.1, 1.1)
		plt.ylim(-1.5, 1.5)
		if i > 0:
			plt.text(0, -1.2, 'ln $\lambda$ = %.0f' % lnL[i], fontsize=12)
		else:
			plt.text(0, -1.2, 'ln $\lambda$ = $-\infty$', fontsize=12)
		plt.scatter(x, t, s=60, c='white', linewidths='2', edgecolors='blue')
		plt.plot(xx, np.sin(2.*np.pi*xx), color='green', linestyle='dashed')
		plt.plot(xx, y(w, xx), color='red')
	
	plt.show()

if __name__ == '__main__':
	main()
