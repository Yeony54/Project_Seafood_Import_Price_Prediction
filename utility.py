# System Libraries
import os
import warnings

# Data handling Libraries
import pandas as pd
import numpy as np

# Visuzliation Libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib import colors
import seaborn as sns

# Traning & Modeling
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import randint

# Date Functions
from datetime import date, timedelta, datetime
from calendar import monthrange


def set_week(df, date):
    """
    dataframe 의 년월일 날짜 컬럼을 년 컬럼과 주차 컬럼으로 분리하는 함수
    :param df: datetime 형식의 컬럼을 가지고 있는 dataframe
    :param date: df에서 datetime 형식을 가진 컬럼명
    :return: date의 연도 컬럼과 주차 컬럼을 추가한 dataframe
    """
    df[date] = pd.to_datetime(df[date])
    df[date] = df[date].dt.date
    df['year'] = df.apply(func=lambda x: x[date].isocalendar()[0], axis=1)
    df['week'] = df.apply(func=lambda x: x[date].isocalendar()[1], axis=1)
    df.drop(date, axis=1, inplace=True)


def check_week(df):
    """
    dataframe에 sdate 과 edate 사이에 모든 데이터가 있는지 확인하는 함수
    :param df: 검사하고자 하는 dataframe (set_week 형태)
    :return: None
    """
    cnt = 0
    sdate = date(2015, 12, 28)  # start date
    edate = date(2019, 12, 30)  # end date
    delta = edate - sdate  # as timedelta
    mem = set()

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        year, week = day.isocalendar()[0], day.isocalendar()[1]
        if year * 100 + week in mem:
            continue
        mem.add(year * 100 + week)
        if df[(df['year'] == year) & (df['week'] == week)].empty:
            print((year, week), end="")
            cnt += 1
    if cnt > 0:
        print()
    print("missing", cnt, "values")

    
# Wrangling Hypothesis Validation Functions
def RMSE(y, y_pred):
    return mean_squared_error(y, y_pred) ** 0.5


def train_model(train_data, target_data, model=LinearRegression()):  # baseline model : LinearRegression
    x_train, x_test, y_train, y_test = train_test_split(train_data, target_data, random_state=0)

    model.fit(x_train, y_train)
    print("Model Training Complete!")

    pred_train, pred_test = model.predict(x_train), model.predict(x_test)

    plt.figure(figsize=(10, 8))
    #     plt.scatter(pred_train, y_train, s=10)
    sns.regplot(pred_train, y_train, color='g')
    plt.xlabel("Predicted price")
    plt.ylabel("Actual price")
    # plt.savefig(os.path.join(root, 'IMAGES', str(cnt) + '.png'), transparent=True)
    plt.show()

    # cvs = cross_val_score(model, x_test, y_test, cv = 5)
    # print(">> cross_val_score mean =", cvs.mean())
    print(">> RMSE train =", RMSE(y_train, pred_train))
    print(">> RMSE validation =", RMSE(y_test, pred_test))
    print(">> MAE train =", mean_absolute_error(pred_train, y_train))
    print(">> MAE validation =", mean_absolute_error(pred_test, y_test))
    print("-------------------------------------------------")

    return model


def print_importance(model, df, added_columns):
    importance = model.coef_
    fs_data = []
    for i, x in enumerate(importance):
        fs_data.append([abs(x), df.columns[i]])
    fs_data.sort(key=lambda x: x[0], reverse=True)

    # 추가한 컬럼의 중요도
    for i in range(len(fs_data)):
        if fs_data[i][1] in added_columns:
            print(fs_data[i][1], ":", fs_data[i][0], ">", i, "순위")
    print("-------------------------------------------------")
    print("총", len(fs_data), "개")

    return fs_data


def model_scaler(data, col, scaler=None):
    '''
    정규화 함수
    data : dataframe
    column : P_PRICE
    scaler : standard, robust, minmax, log

    '''

    features = data.drop(col, axis=1)
    target = data[col]

    if scaler == 'standard':
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        return features, target

    elif scaler == 'robust':
        scaler = RobustScaler()
        features = scaler.fit_transform(features)

        return features, target

    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

        return features, target

    elif scaler == 'log':
        features = np.log1p(features)

        return features, target

    elif scaler == 'None':

        return features, target


################################################################################################################################################

def model_train(data, col, scaler, cv=5, n_iter=50, model=None):
    '''

    data : dataframe
    column : P_PRICE
    scaler : standard, robust, minmax, log
    model_name : linear, ridge, lasso, elastic, decisiontree,
                 randomforest, ada, gradient, xgb, lgbm

    '''

    features, target = model_scaler(data, col, scaler)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    if model == 'linear':

        model = LinearRegression()
        neg_mse_scores = cross_val_score(model, features, target, scoring='neg_mean_squared_error', cv=cv)
        rmse_scores = np.sqrt(-1 * neg_mse_scores)
        avg_rmse = np.mean(rmse_scores)

        print('RMSE : {:.4f}'.format(avg_rmse))

    elif model == 'ridge':

        params = {
            'alpha': (
            0.01, 0.0001, 0.003, 0.5, 0.04, 0.008, 0.912, 0.098, 0.0625, 0.1763, 0.001, 0.351, 0.096, 0.853, 0.185,
            0.01825, 0.012385, 0.1, 1, 10, 100, 200, 50, 30, 20, 29, 58),
            'fit_intercept': (True, False),
            'normalize': (True, False),

        }

        ridge = Ridge(random_state=0)
        final = RandomizedSearchCV(ridge, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(x_train, y_train)
        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))

    elif model == 'lasso':

        params = {
            'alpha': (
            0.01, 0.0001, 0.003, 0.5, 0.04, 0.1734, 0.098, 0.0074, 0.0001, 0.00923, 0.98, 0.174, 0.008, 0.001, 0.351,
            0.096, 0.853, 0.185, 0.01825, 0.012385, 0.1, 1, 10, 100, 200, 50, 30, 20, 29, 58),
            'fit_intercept': (True, False),
            'normalize': (True, False),

        }

        lasso = Lasso(random_state=0)
        final = RandomizedSearchCV(lasso, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)
        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))

    elif model == 'elastic':

        params = {
            'alpha': (0.1, 0.01, 0.5, 1, 3, 5, 10),
            'l1_ratio': (0.01, 0.0001, 0.003, 0.5, 0.04, 0.008, 0.1, 0.0125, 0.98263, 0.0935)
        }

        elastic = ElasticNet()
        final = RandomizedSearchCV(elastic, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)
        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))
    elif model == 'decisiontree':

        params = {
            'max_depth': randint(10, 1000),
            # 'min_child_samples': randint(5, 50),
            'min_samples_split': randint(1, 1000),
            'min_samples_leaf': randint(1, 1000),

        }

        dt = DecisionTreeRegressor(random_state=0)
        final = RandomizedSearchCV(dt, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)
        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))
    elif model == 'randomforest':

        params = {
            'max_depth': randint(1, 5000),
            'n_estimators': randint(1, 5000),
            # 'min_child_samples': randint(5, 50),
            'min_samples_leaf': randint(1, 5000),
            'min_samples_split': randint(1, 5000),
            'max_leaf_nodes': randint(1, 5000)

        }

        rf = RandomForestRegressor(random_state=0)
        final = RandomizedSearchCV(rf, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)
        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))


    elif model == 'gradinet':

        params = {'n_estimators': randint(30, 1000),
                  'learning_rate': (
                  0.01, 0.0001, 0.003, 0.5, 0.04, 0.008, 0.001, 0.351, 0.096, 0.853, 0.185, 0.01825, 0.012385, 0.1),
                  'subsample': (0.01, 0.1, 0.5, 0.08, 0.35, 0.3, 0.001, 0.03, 0.006, 0.153, 0.193, 0.0012, 0.0083, 1),
                  'min_samples_split': randint(1, 5000),
                  'max_depth': randint(1, 5000),
                  }

        grad = GradientBoostingRegressor()
        final = RandomizedSearchCV(grad, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)
        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))

    elif model == 'xgb':

        params = {'n_estimators': randint(1, 5000),
                  'learning_rate': (
                  0.01, 0.0001, 0.003, 0.5, 0.04, 0.008, 0.001, 0.351, 0.096, 0.853, 0.185, 0.01825, 0.012385, 0.1),
                  'max_depth': randint(1, 1000),
                  'min_child_weight': randint(1, 5000),
                  }

        xgb = XGBRegressor()
        final = RandomizedSearchCV(xgb, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)

        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))

    elif model == 'lgbm':
        params = {'n_estimators': randint(1, 5000),
                  'learning_rate': (
                  0.01, 0.0001, 0.003, 0.5, 0.04, 0.008, 0.001, 0.351, 0.096, 0.853, 0.185, 0.01825, 0.012385, 0.1),
                  'max_depth': randint(-1, 10),
                  'min_child_weight': (0.001, 0.01, 0.5, 0.005, 0.0038, 0.001856, 0.0811, 0.1, 0.0931, 0.9, 1),
                  'num_leaves': randint(3, 5000),
                  'min_child_samples': randint(1, 5000)
                  }

        lgbm = LGBMRegressor()
        final = RandomizedSearchCV(lgbm, param_distributions=params, cv=cv, scoring='neg_mean_squared_error',
                                   n_iter=n_iter, n_jobs=-1, random_state=0)
        final.fit(features, target)

        pred = final.predict(x_test)

        print('Best Params:', final.best_params_)
        print('Best Score:', np.sqrt(-1 * final.best_score_))
        print('Predict RMSE:', (np.sqrt(mean_squared_error(y_test, pred))))