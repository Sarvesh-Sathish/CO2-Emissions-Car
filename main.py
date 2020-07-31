import matplotlib.pyplot as plt
import pandas as pd
import pylab as pb
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumptionCo2.csv")

data_top = df.head()
data_top

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='red')
plt.xlabel("Engine size")
plt.ylabel("Emission")

plt.savefig('EngineSize-vs-Emission')
#supposing that 'ENGINESIZE' is the only original feature set
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)

print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
XX = np.arange(0.0,10.0,0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2) + clf.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy , '-r')
plt.xlabel('Engine Size')
plt.ylabel('Emission')
plt.savefig('EngineSize-vs-CO2Emissions.png')
