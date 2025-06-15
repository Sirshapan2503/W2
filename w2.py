
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


train = pd.read_csv('c:/Users/sirsh/OneDrive/Desktop/CODING/hacktrain.csv')
test = pd.read_csv('c:/Users/sirsh/OneDrive/Desktop/CODING/hacktest.csv')


ndvi_cols = [col for col in train.columns if '_N' in col]


X_train_raw = train[ndvi_cols]
y_train = train['class']
X_test_raw = test[ndvi_cols]


imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(multi_class='multinomial',
                           solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


submission = pd.DataFrame({
    'ID': test['ID'],
    'class': y_pred
})
submission.to_csv('submission.csv', index=False)


print("✅ submission.csv saved and ready for upload.")
