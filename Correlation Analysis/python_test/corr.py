import numpy as np

a = np.array([[0.3,0.41, 0.75],[0.2,0.31, 0.15]])
b = np.array([[0.1,0.99,0.23],[0.1,0.39,0.43]])
c1 = np.corrcoef(a,a,rowvar=False)
c2 = np.corrcoef(a,a,rowvar=True)
ii = 42
