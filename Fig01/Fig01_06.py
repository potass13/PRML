# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	M = 9
	N = [15, 100]
	sigma = 0.3
	
	xx = np.linspace(0, 1, 200)
	
	for i, n in enumerate(N):
		x = np.linspace(0, 1, n)
		t = np.sin(2.*np.pi*x)+np.random.normal(0, sigma, len(x))
		df = pd.DataFrame(np.array([x, t]).transpose(), columns=['x', 't'])
		y = np.poly1d(np.polyfit(df.x.values, df.t.values, M))
		
		plt.subplot(1, 2, i+1)
		plt.xlabel('x')
		plt.ylabel('t')
		plt.xlim(-0.1, 1.1)
		plt.ylim(-1.5, 1.5)
		plt.text(0.8, 1.0, 'N = %d' % n, fontsize=15)
		
		plt.scatter(x, t, s=60, c='white', linewidths='2', edgecolors='blue')
		plt.plot(xx, np.sin(2.*np.pi*xx), color='green', linestyle='dashed')
		plt.plot(xx, y(xx), color='red')
	
	plt.show()

if __name__ == '__main__':
	main()
