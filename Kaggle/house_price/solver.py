from warnings import simplefilter
import numpy as np
import pandas as pd
from pyearth import Earth
from sklearn.feature_selection import RFE
simplefilter("ignore")

train_data = (pd.read_csv('C:\\Users\\hongj\\Desktop\\kaggle\\house_price\\train.csv')).drop('Id', axis=1)
train_Y = np.log(np.array(train_data['SalePrice']))

train_data = train_data.drop('SalePrice', axis=1)
test_data = pd.read_csv('C:\\Users\\hongj\\Desktop\\kaggle\\house_price\\test.csv')
test_index = test_data['Id']
test_data = test_data.drop('Id', axis=1)

total_data = pd.concat([train_data, test_data]).reset_index(drop=True)

total_key = set(total_data.columns)
numeric_key = set(total_data._get_numeric_data().columns)
category_key = total_key - numeric_key

total_numeric_data = total_data[list(numeric_key)]
total_category_data = pd.get_dummies(total_data[list(category_key)])

total_data = pd.concat([total_category_data,
                        total_numeric_data.clip(total_numeric_data.quantile(0.01).to_dict(),
                                                total_numeric_data.quantile(0.99).to_dict(),
                                                axis=1)],
                       axis=1)
print(total_data.shape)
total_data = total_data.fillna(total_data.mean())

print(total_data.head(5))

train_data = total_data[total_data.index < 1460]
test_data = total_data[total_data.index >= 1460]

rfe = RFE(Earth(), step=15, verbose=2).fit(train_data, train_Y)
validKeys = list(train_data.columns[rfe.support_])

train_data = train_data[validKeys]
test_data = test_data[validKeys]

model = Earth().fit(train_data, train_Y)
predict = model.predict(test_data)
predict = np.exp(predict)

submission = pd.DataFrame()
submission['Id'] = test_index
submission['SalePrice'] = predict
submission.to_csv("C:\\Users\\hongj\\Desktop\\kaggle\\house_price\\submission.csv", index=False)
