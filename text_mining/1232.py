import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(100,10))
x = np.mat(2)
y = np.mat(4)
mlp.fit(x,y)
print(mlp.predict(2))
print(mlp.coefs_)