# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alpha = 5.0e-03
beta = 11.1

def makems(m, dataset):
	Phi = np.array([dataset.x.values**k for k in range(0, m+1)])
	inv_matS = alpha*np.eye(m+1)+beta*np.dot(Phi, Phi.T)
	matS = np.linalg.inv(inv_matS)
	
	def mean(xx):
		phi_xx = np.array([xx**k for k in range(0, m+1)])
		return beta*np.dot(np.dot(phi_xx.T, matS), np.sum(Phi*dataset.t.values, axis=1))
	
	def var(xx):
		phi_xx = np.array([xx**k for k in range(0, m+1)])
		return (1./beta+np.dot(np.dot(phi_xx.T, matS), phi_xx)).diagonal()
		
	return mean, var

def main():
	M = 9
	N = [4, 5, 10, 100]
	sigma = 1./np.sqrt(beta)
	
	for index, n in enumerate(N):
		x = np.linspace(0, 1, n)
		t = np.sin(2.*np.pi*x)+np.random.normal(0, sigma, len(x))
		dataset = pd.DataFrame(np.array([x, t]).T, columns=['x', 't'])
	
		mean, var = makems(M, dataset)
		xx = np.linspace(0, 1, 100)
		m = np.array(mean(xx))
		s = np.sqrt(np.array(var(xx)))
		
		plt.subplot(2, 2, index+1)
		plt.xlabel('x')
		plt.ylabel('t')
		plt.xlim(-0.1, 1.1)
		plt.ylim(-1.5, 1.5)
		plt.text(0.8, 1.0, 'N = %d' % n, fontsize='12')
		plt.scatter(x, t, s=60, c='white', linewidths='2', edgecolors='blue')
		plt.plot(xx, np.sin(2.*np.pi*xx), color='green', linestyle='dashed')
		plt.plot(xx, m, color='red')
		plt.fill_between(xx, m-s, m+s, facecolor='red', alpha=0.3)
	
	plt.show()

if __name__ == '__main__':
	main()
