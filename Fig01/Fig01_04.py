# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def LeastRootSq(x, t, M):
	aaa = np.array([x**k for k in range(2*M+1)])
	aa = np.sum(aaa, axis=1)
	A = np.array([[aa[i+j] for i in range(M+1)] for j in range(M+1)])
	
	tt = np.array([x**k for k in range(M+1)])
	T = np.sum(tt*t, axis=1)
	
	w = np.linalg.solve(A, T)
	f = np.poly1d(w[::-1])
	
	return(f, w)
	
def main():
	N = 10
	M = [0, 1, 3, 9]
	
	xx = np.linspace(0, 1, 100)
	
	x = np.linspace(0, 1, N)
	t = np.sin(2.*np.pi*x)+np.random.normal(0, 0.2, len(x))
	
	for i,m in enumerate(M):
		f, w = LeastRootSq(x, t, m)
		
		plt.subplot(2,2,i+1)
		plt.xlabel('x')
		plt.ylabel('t')
		plt.xlim(-0.1, 1.1)
		plt.ylim(-1.5, 1.5)
		plt.text(0.8, 1.0, 'M = %d' % m, fontsize=15)
		plt.scatter(x, t, s=60, c='white', linewidths='2', edgecolors='blue')
		plt.plot(xx, np.sin(2.*np.pi*xx), color='green', linestyle='dashed')
		plt.plot(xx, f(xx), color='red')
	
	plt.show()

if __name__ == '__main__':
	main()
