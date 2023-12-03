from sklearn.model_selection import  KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import r2_score

def cross_validate (model, X, y):
    cv = KFold(n_splits = 10)
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

    rmse = []
    nrmse = []
    r2 = []
    predictions = []
    
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        
        print("Fold:", i + 1, end = "\r")
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        rmse.append(np.sqrt(((y_test - pred) ** 2).mean()))
        nrmse.append(np.sqrt(((y_test - pred) ** 2).mean()) / y_test.mean(axis = 0))
        r2.append(1 - (1 - r2_score(y_test, pred))*(len(X) - 1) / (len(X) - len(X.columns) - 1))
        predictions.append(pred)

    print("Mean RMSE:", np.mean(np.array(rmse), axis = 0))
    print("Mean NRMSE:", np.mean(np.array(nrmse), axis = 0))
    print("Mean Adjusted R^2:", np.mean(np.array(r2), axis = 0))

    return np.hstack(predictions)

def cross_validate_spatial (model, X, y, coords):
    
    cv = KFold(n_splits = 10)
    scaler = StandardScaler()

    rmse = []
    nrmse = []
    r2 = []
    predictions = []

    for i, (train_index, test_index) in enumerate(cv.split(X)):
        
        print("Fold:", i + 1, end = "\r")
        X_train, X_test, y_train, y_test, coords_train, coords_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index], coords.iloc[train_index], coords.iloc[test_index]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train, coords_train)
        pred = model.predict(X_test, coords_test)

        rmse.append(np.sqrt(((y_test - pred) ** 2).mean()))
        nrmse.append(np.sqrt(((y_test - pred) ** 2).mean()) / y_train.mean(axis = 0))
        r2.append(1 - (1 - r2_score(y_test, pred))*(len(X) - 1) / (len(X) - len(X.columns) - 1))

        predictions.append(pred)

    print("Mean RMSE:", np.mean(np.array(rmse), axis = 0))
    print("Mean NRMSE:", np.mean(np.array(nrmse), axis = 0))
    print("Mean Adjusted R^2:", np.mean(np.array(r2), axis = 0))

    return np.hstack(predictions)