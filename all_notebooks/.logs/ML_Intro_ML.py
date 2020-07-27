%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Fri, 10 Jul 2020 16:35:15
%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Fri, 10 Jul 2020 16:35:17
import numpy as np
import matplotlib.pyplot as plt# Fri, 10 Jul 2020 16:35:24
%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
import numpy as np
import matplotlib.pyplot as plt
%logstop
%logstart -rtq ~/.logs/ML_Intro_ML.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Fri, 10 Jul 2020 16:35:24
import numpy as np
import matplotlib.pyplot as plt# Fri, 10 Jul 2020 16:35:24
X = np.linspace(0, 1, 100)
exp = np.random.choice([2, 3])
y = X**exp + np.random.randn(X.shape[0])/10
plt.plot(X, y, '.');# Fri, 10 Jul 2020 16:35:24
p = np.polyfit(X, y, 1)
z = np.poly1d(p)
plt.plot(X, y, '.')
plt.plot(X, z(X), label=r"Model: ${:.2f}x + {:.2f}$".format(*p))
plt.plot(X, X**exp, label=r'Truth: $x^{}$'.format(exp))
plt.legend();# Fri, 10 Jul 2020 16:35:24
X = np.linspace(0, 2, 100)
y = X**exp + np.random.randn(X.shape[0])/10
plt.plot(X, z(X), label=r"${:.2f}x + {:.2f}$".format(*p))
plt.plot(X, y,'.', label=r'$x^{}$'.format(exp))
plt.legend();# Fri, 10 Jul 2020 16:35:25
p = np.polyfit(X, y, 15)
z = np.poly1d(p)
plt.figure(figsize=[14, 6])
plt.plot(X, z(X), label=r"${:.2f}x^{{15}} + {:.2f}x^{{14}} + ... + {:.2f}$".format(*p[[0, 1, -1]]))
plt.plot(X, y,'.', label=r'$x^{}$'.format(exp))
plt.legend();# Fri, 10 Jul 2020 16:35:25
X = np.linspace(0, 2.5, 100)
y = X**exp + np.random.randn(X.shape[0])/10
plt.plot(X, z(X), label=r"model")
plt.plot(X, y,'.', label=r'$x^{}$'.format(exp))
plt.legend();# Fri, 10 Jul 2020 16:35:25
from sklearn.linear_model import LinearRegression# Fri, 10 Jul 2020 16:35:25
lr = LinearRegression(fit_intercept=True, normalize=False)
lr# Fri, 10 Jul 2020 16:35:25
lr.fit(X.reshape(-1, 1), y)# Fri, 10 Jul 2020 16:35:25
lr.coef_, lr.intercept_# Fri, 10 Jul 2020 16:35:25
predictions = lr.predict(X.reshape(-1, 1))
plt.plot(X, y, '.', label='data')
plt.plot(X, predictions, label='model')
plt.legend();# Fri, 10 Jul 2020 16:35:26
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = Pipeline([
    ('polynomial_transform', PolynomialFeatures(3)),
    ('linear_fit', LinearRegression())
])

pipe.fit(X.reshape(-1, 1), y)

predictions = pipe.predict(X.reshape(-1, 1))
plt.plot(X, y, '.', label='data')
plt.plot(X, predictions, label='model')
plt.legend();# Fri, 10 Jul 2020 16:35:26
X = np.linspace(0, 4, 100)
y = X**exp + np.random.randn(X.shape[0])/10
predictions = pipe.predict(X.reshape(-1, 1))
plt.plot(X, y, '.', label='data')
plt.plot(X, predictions, label='model')
plt.legend();