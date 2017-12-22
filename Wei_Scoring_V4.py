import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import math 

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble

from sklearn.externals import joblib

import json

import warnings
warnings.filterwarnings('ignore')

def create_time_feature(df):       
    df['created_at_datetime'] = df['created_at'].astype("datetime64[s]")
    df['created_at_year'], df['created_at_month'], df['created_at_day'], df['created_at_date'], df['created_at_dayOfWeek'], df['created_at_time'], df['created_at_hour'], df['created_at_minute'], df['created_at_second'] = df['created_at_datetime'].dt.year, df['created_at_datetime'].dt.month, df['created_at_datetime'].dt.day, df['created_at_datetime'].dt.date, df['created_at_datetime'].dt.dayofweek, df['created_at_datetime'].dt.time, df['created_at_datetime'].dt.hour, df['created_at_datetime'].dt.minute, df['created_at_datetime'].dt.second
    df.loc[df['created_at_dayOfWeek'].isin([5, 6]), 'created_at_isWeekend'] = 1
    df.loc[df['created_at_dayOfWeek'].isin([0, 1, 2, 3, 4]), 'created_at_isWeekend'] = 0
    cal = calendar()
    holidays = cal.holidays(start=df['created_at_date'].min(), end=df['created_at_date'].max())
    df['created_at_isHoliday'] = np.where(df.created_at_datetime.dt.normalize().isin(holidays), 1, 0)
    return df