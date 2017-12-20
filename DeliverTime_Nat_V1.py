# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:07:56 2017

@author: ds1
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Train = pd.read_csv(r'C:\Users\ds1\Downloads\DoorDash\Data Science\historical_data.csv')
Train.info()
Train.head()

'''
market_id                                       196441 non-null float64
created_at                                      197428 non-null object
actual_delivery_time                            197421 non-null object
store_id                                        197428 non-null object
store_primary_category                          192668 non-null object
order_protocol                                  196433 non-null float64
total_items                                     197428 non-null int64
subtotal                                        197428 non-null int64
num_distinct_items                              197428 non-null int64
min_item_price                                  197428 non-null int64
max_item_price                                  197428 non-null int64
total_onshift_dashers                           181166 non-null float64
total_busy_dashers                              181166 non-null float64
total_outstanding_orders                        181166 non-null float64
estimated_order_place_duration                  197428 non-null int64
estimated_store_to_consumer_driving_duration    196902 non-null float64
'''

'''
Data Processing:

    Calcualte True Label
        time diff
    Time Feature:
        Day, Hour, Weekend, IsHoliday,
    Weather feature
        Download
        Merge
    Categorical Features
        encoding
        re-grouping
    Continuous Featurs
        direct use
        binning
    Missing Values
        Categorical: UNK group
        Continuous: imputing (mean, medain, mode)
    Store_Id
        Hashing

'''

'''
Time: 
    Calcualte True Label
        time diff
    Time Feature:
        Day, Hour, Weekend, IsHoliday, etc
'''

#subset data: removed nan actual_delivery_time, since these do not have labels, removed 7 rows
Train = Train[pd.notnull(Train['actual_delivery_time'])]

#conver time object to datetime64
Train['created_at_datetime'] = Train['created_at'].astype("datetime64[s]")
Train['actual_delivery_time_datetime'] = Train['actual_delivery_time'].astype("datetime64[s]")

#create label: duration
Train['duration'] = Train['actual_delivery_time_datetime'] - Train['created_at_datetime']
Train['duration'] = Train['duration'] / np.timedelta64(1, 's')

#Duration description
Train.duration.describe()
#maybe remove outlier: max value
#count    1.974210e+05
#mean     2.908257e+03
#std      1.922961e+04
#min      1.010000e+02
#25%      2.104000e+03
#50%      2.660000e+03
#75%      3.381000e+03
#max      8.516859e+06

#remove outlier: remove more than 7200 seconds, which is 2 hrs, and that removed 1089 obs
#and left with  0.9944838415560734 of the raw data 
Train = Train.loc[Train['duration'] < 7200]

#create created_at_year, created_at_month, created_at_day, created_at_date, created_at_dayOfWeek, 
#created_at_time, created_at_hour, created_at_minute, created_at_second, created_at_isWeekend,
#created_at_isHoliday 
Train['created_at_year'], Train['created_at_month'], Train['created_at_day'], Train['created_at_date'], Train['created_at_dayOfWeek'], Train['created_at_time'], Train['created_at_hour'], Train['created_at_minute'],  Train['created_at_second'] = Train['created_at_datetime'].dt.year, Train['created_at_datetime'].dt.month, Train['created_at_datetime'].dt.day, Train['created_at_datetime'].dt.date, Train['created_at_datetime'].dt.dayofweek, Train['created_at_datetime'].dt.time, Train['created_at_datetime'].dt.hour, Train['created_at_datetime'].dt.minute, Train['created_at_datetime'].dt.second

Train.loc[Train['created_at_dayOfWeek'].isin([5, 6]), 'created_at_isWeekend'] = 1
Train.loc[Train['created_at_dayOfWeek'].isin([0, 1, 2, 3, 4]), 'created_at_isWeekend'] = 0

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays(start=Train['created_at_date'].min(), end=Train['created_at_date'].max())
Train['created_at_isHoliday'] = np.where(Train.created_at_datetime.dt.normalize().isin(holidays), 1, 0)

#time features description

#daily counts of 
TimeSeriesTrain = pd.DataFrame(Train['created_at_date'].value_counts().reset_index().rename(columns={'index': 'start_date', 0: 'Count'}))
TimeSeriesTrain.columns = ['created_at_date', 'Count']
TimeSeriesTrain = TimeSeriesTrain.sort_values(by='created_at_date', ascending=True)

plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 22})
plt.plot(TimeSeriesTrain['created_at_date'], TimeSeriesTrain['Count'])
plt.tight_layout()

TimeSeriesTrain['Count'].describe()
#count      30.000000   #only 30 days of data
#mean     6580.700000
#std      1952.747987
#min         1.000000
#25%      6048.000000
#50%      6739.000000
#75%      7878.750000
#max      9149.000000

TimeSeriesTrain.loc[TimeSeriesTrain['Count'] == 1]['created_at_date']  # 2014-10-19
TimeSeriesTrain.loc[TimeSeriesTrain['Count'] == 9149]['created_at_date']  # 2015-02-07

Train['created_at_year'].value_counts()/Train['created_at_year'].value_counts().sum()
#2015    197420
#2014         1

Train['created_at_date'].value_counts()/Train['created_at_date'].value_counts().sum()
#only has 30 days of data, may want to consider dropping 2014 data
#2015-02-07    9149
#2015-02-15    9085
#2015-02-14    9016
#2015-02-08    8873
#2015-01-24    8230
#2015-01-31    8146
#2015-01-25    7934
#2015-02-16    7931
#2015-02-01    7722
#2015-02-13    7383
#2015-02-09    7278
#2015-02-06    7118
#2015-02-05    6850
#2015-02-12    6784
#2015-01-30    6763
#2015-02-02    6715
#2015-01-23    6611
#2015-02-11    6403
#2015-02-03    6369
#2015-02-04    6113
#2015-02-10    6090
#2015-02-17    6078
#2015-01-29    6038
#2015-01-22    6001
#2015-01-28    5669
#2015-01-27    5524
#2015-01-26    5478
#2015-02-18    3981
#2015-01-21    2088
#2014-10-19       1

Train['created_at_month'].value_counts()/Train['created_at_month'].value_counts().sum()
#2     128938
#1      68482
#10         1

Train['created_at_dayOfWeek'].value_counts()/Train['created_at_dayOfWeek'].value_counts().sum()
#5 and 6 are weekends, which have more orders
#5    0.174961
#6    0.170271
#4    0.141196
#0    0.138800
#3    0.130042
#2    0.122854
#1    0.121877

Train['created_at_isWeekend'].value_counts()/Train['created_at_isWeekend'].value_counts().sum()
#0.0    0.654768
#1.0    0.345232

Train['created_at_hour'].value_counts()/Train['created_at_hour'].value_counts().sum()
#2     0.187280
#1     0.142776
#3     0.137108
#20    0.078816
#4     0.077246
#19    0.068589
#0     0.064173
#21    0.058069
#22    0.044681
#23    0.041348
#5     0.035943
#18    0.025833
#17    0.017288
#16    0.010683
#6     0.007172
#15    0.002725
#14    0.000203
#7     0.000056
#8     0.000010

'''
Continuous Featurs
    direct use
    missing data imputation
    binning
'''
#total_items
#remove outliers: removed greater than 20 items, 414 obs
Train['total_items'].value_counts(dropna = False)/Train['total_items'].value_counts(dropna = False).sum()
Train['total_items'].describe()
plt.hist(Train['total_items'], bins = 50)
len(Train.loc[Train['total_items'] >= 20]) #414
Train = Train.loc[Train['total_items'] < 20]

#subtotal
#remove outliers: removed greater than 12000 subtotal, 456 obs
Train['subtotal'].value_counts(dropna = False)/Train['subtotal'].value_counts(dropna = False).sum()
Train['subtotal'].describe()
plt.hist(Train['subtotal'], bins = 50)
len(Train.loc[Train['subtotal'] >= 12000]) #456
Train = Train.loc[Train['subtotal'] < 12000]

#num_distinct_items
Train['num_distinct_items'].value_counts(dropna = False)/Train['num_distinct_items'].value_counts(dropna = False).sum()
Train['num_distinct_items'].describe()
plt.hist(Train['num_distinct_items'], bins = 50)

#min_item_price
#remove outliers: removed smaller than 0 min_item_price, 10 obs
#remove outliers: removed greater than 5000 min_item_price, 67 obs
Train['min_item_price'].value_counts(dropna = False)/Train['min_item_price'].value_counts(dropna = False).sum()
Train['min_item_price'].describe()
plt.hist(Train['min_item_price'], bins = 50)
len(Train.loc[Train['min_item_price'] < 0]) #10
Train = Train.loc[Train['min_item_price'] >= 0]
len(Train.loc[Train['min_item_price'] > 5000]) #67
Train = Train.loc[Train['min_item_price'] <= 5000]

#max_item_price
#remove outliers: removed greater than 5000 max_item_price, 98 obs
Train['max_item_price'].value_counts(dropna = False)/Train['max_item_price'].value_counts(dropna = False).sum()
Train['max_item_price'].describe()
plt.hist(Train['max_item_price'], bins = 50)
len(Train.loc[Train['max_item_price'] > 5000]) #98
Train = Train.loc[Train['max_item_price'] <= 5000]

#total_onshift_dashers
#fill nan with mean
#replace negative values with mean  
Train['total_onshift_dashers'].value_counts(dropna = False)/Train['total_onshift_dashers'].value_counts(dropna = False).sum()
Train['total_onshift_dashers'].describe()
plt.hist(Train['total_onshift_dashers'][~np.isnan(Train['total_onshift_dashers'])])
Train['total_onshift_dashers'] = Train['total_onshift_dashers'].fillna(int(Train['total_onshift_dashers'].mean()))
Train['total_onshift_dashers'].loc[Train['total_onshift_dashers'] <0] = int(Train['total_onshift_dashers'].mean())

#total_busy_dashers
#fill nan with mean
#replace negative values with mean  
Train['total_busy_dashers'].value_counts(dropna = False)/Train['total_busy_dashers'].value_counts(dropna = False).sum()
Train['total_busy_dashers'].describe()
plt.hist(Train['total_busy_dashers'][~np.isnan(Train['total_busy_dashers'])])
Train['total_busy_dashers'] = Train['total_busy_dashers'].fillna(int(Train['total_busy_dashers'].mean()))
Train['total_busy_dashers'].loc[Train['total_busy_dashers'] <0] = int(Train['total_busy_dashers'].mean())

#total_outstanding_orders
#fill nan with mean
#replace negative values with mean 
Train['total_outstanding_orders'].value_counts(dropna = False)/Train['total_outstanding_orders'].value_counts(dropna = False).sum()
Train['total_outstanding_orders'].describe()
plt.hist(Train['total_outstanding_orders'][~np.isnan(Train['total_outstanding_orders'])])
Train['total_outstanding_orders'] = Train['total_outstanding_orders'].fillna(int(Train['total_outstanding_orders'].mean()))
Train['total_outstanding_orders'].loc[Train['total_outstanding_orders'] <0] = int(Train['total_outstanding_orders'].mean())

#estimated_order_place_duration
#only 2 large groups, all others are pretty small, rebin to 3 groups
Train['estimated_order_place_duration'].value_counts(dropna = False)/Train['estimated_order_place_duration'].value_counts(dropna = False).sum()
Train['estimated_order_place_duration'].describe()
plt.hist(Train['estimated_order_place_duration'], bin=50)
Train['estimated_order_place_duration_rebinned'] = Train['estimated_order_place_duration']
Train['estimated_order_place_duration_rebinned'][Train['estimated_order_place_duration'] == 251] = 'fast'
Train['estimated_order_place_duration_rebinned'][Train['estimated_order_place_duration'] == 446] = 'slow'
Train['estimated_order_place_duration_rebinned'][(Train['estimated_order_place_duration'] != 251) & (Train['estimated_order_place_duration'] != 446)] = 'other'
Train['estimated_order_place_duration_rebinned'].value_counts(dropna = False)

#estimated_store_to_consumer_driving_duration
#fill nan with mean
Train['estimated_store_to_consumer_driving_duration'].value_counts(dropna = False)/Train['estimated_store_to_consumer_driving_duration'].value_counts(dropna = False).sum()
Train['estimated_store_to_consumer_driving_duration'].describe()
plt.hist(Train['estimated_store_to_consumer_driving_duration'][~np.isnan(Train['estimated_store_to_consumer_driving_duration'])])
Train['estimated_store_to_consumer_driving_duration'] = Train['estimated_store_to_consumer_driving_duration'].fillna(int(Train['estimated_store_to_consumer_driving_duration'].mean()))


''' 
Categorical Features
   encoding
   re-grouping

market_id                                       196441 non-null float64
created_at                                      197428 non-null object
actual_delivery_time                            197421 non-null object
store_id                                        197428 non-null object
store_primary_category                          192668 non-null object
order_protocol                                  196433 non-null float64
total_items                                     197428 non-null int64
subtotal                                        197428 non-null int64
num_distinct_items                              197428 non-null int64
min_item_price                                  197428 non-null int64
max_item_price                                  197428 non-null int64
total_onshift_dashers                           181166 non-null float64
total_busy_dashers                              181166 non-null float64
total_outstanding_orders                        181166 non-null float64
estimated_order_place_duration                  197428 non-null int64
estimated_store_to_consumer_driving_duration    196902 non-null float64
'''   
#market_id: fill Nan as 0
Train['market_id'].value_counts(dropna= False)/Train['market_id'].value_counts(dropna= False).sum()
Train['market_id'][Train['market_id'].isnull()] = 0

#store_id
#very imbalanced, create binning 
Train['store_id'].value_counts(dropna= False)/Train['store_id'].value_counts(dropna= False).sum()

store_counts_df = pd.DataFrame(Train['store_id'].value_counts().reset_index().rename(columns={'index': 'store_id', 0: 'store_id_count'}))
store_counts_df.columns = ['store_id', 'store_id_count']
store_counts_df = store_counts_df.sort_values(by='store_id', ascending=True)

plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 22})
plt.plot(store_counts_df['store_id'], store_counts_df['store_id_count'])
plt.tight_layout()

store_counts_df['store_id_count'].describe()

Train = pd.merge(Train, store_counts_df, on='store_id', how='left')
Train['store_id_rebinned'] = Train['store_id']
Train['store_id_rebinned'][(Train['store_id_count'] <500) & (Train['store_id_count'] >= 400)] = '[400, 500)'
Train['store_id_rebinned'][(Train['store_id_count'] <400) & (Train['store_id_count'] >= 200)] = '[200, 400)'
Train['store_id_rebinned'][(Train['store_id_count'] <200) & (Train['store_id_count'] >= 50)] = '[50, 200)'
Train['store_id_rebinned'][Train['store_id_count'] <50] = '[0, 50)'
Train['store_id_rebinned'].value_counts(dropna= False)
 
#store_primary_category
#very imbalanced, create rebinning
#impute Nan as Unknown
Train['store_primary_category'].value_counts(dropna= False)/Train['store_primary_category'].value_counts(dropna= False).sum()

store_primary_category_counts_df = pd.DataFrame(Train['store_primary_category'].value_counts().reset_index().rename(columns={'index': 'store_primary_category', 0: 'store_primary_category_count'}))
store_primary_category_counts_df.columns = ['store_primary_category', 'store_primary_category_count']
store_primary_category_counts_df = store_primary_category_counts_df.sort_values(by='store_primary_category', ascending=True)

Train = pd.merge(Train, store_primary_category_counts_df, on='store_primary_category', how='left')
Train['store_primary_category_rebinned'] = Train['store_primary_category']
Train['store_primary_category_rebinned'][Train['store_primary_category_rebinned'].isnull()] = 'Unknown'

Train['store_primary_category_rebinned'][(Train['store_primary_category_count'] <3000) & (Train['store_primary_category_count'] >= 2000)] = '[2000, 3000)'
Train['store_primary_category_rebinned'][(Train['store_primary_category_count'] <2000) & (Train['store_primary_category_count'] >= 1000)] = '[1000, 2000)'
Train['store_primary_category_rebinned'][(Train['store_primary_category_count'] <1000) & (Train['store_primary_category_count'] >= 200)] = '[200, 1000)'
Train['store_primary_category_rebinned'][Train['store_primary_category_count'] <200] = '[0, 200)'
Train['store_primary_category_rebinned'].value_counts(dropna= False)
 
#order_protocol
#rebin low freq together as level 0
Train['order_protocol'].value_counts(dropna= False)/Train['order_protocol'].value_counts(dropna= False).sum()
Train['order_protocol'][Train['order_protocol'].isnull()] = 0
Train['order_protocol'].loc[Train['order_protocol'] == 6] = 0
Train['order_protocol'].loc[Train['order_protocol'] == 7] = 0


import sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import math  

TrainFeatures = Train[['duration', 'market_id', 'store_id_rebinned', 'store_primary_category_rebinned',  'order_protocol',  'total_items', 'subtotal', 'num_distinct_items', 'min_item_price',  'max_item_price','total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_store_to_consumer_driving_duration', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour', 'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']]
            
TrainFeatures[['market_id', 'store_id_rebinned', 'store_primary_category_rebinned', 'order_protocol', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour',  'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']] = TrainFeatures[['market_id', 'store_id_rebinned', 'store_primary_category_rebinned', 'order_protocol', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour',  'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']].astype(object)

NumFeatures = ['total_items', 'subtotal', 'num_distinct_items', 'min_item_price',  'max_item_price','total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_store_to_consumer_driving_duration']

CatFeatures = ['market_id', 'store_id_rebinned', 'store_primary_category_rebinned',  'order_protocol', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour', 'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']

X_train, X_test, Y_train, Y_test= train_test_split(TrainFeatures, Train['duration'], test_size=0.4, random_state=2017)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))
#117171 78115 117171 78115
X_train.shape # (117171, 20)

#baseline model: regression only use numeric features
lm = linear_model.LinearRegression()
lm.fit(X_train[NumFeatures], Y_train)
lm_predict_train = lm.predict(X_train[NumFeatures])
lm_predict_test = lm.predict(X_test[NumFeatures])
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, lm_predict_train)))
#Root mean squared error for train 896.74
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, lm_predict_test)))
#Root mean squared error for test: 893.57

#for plot of any model
plt.scatter(Y_test, lm_predict_test)
plt.xlabel('duration actual')
plt.ylabel('duration predicted')
plt.title('duration actual vs predicted')
plt.show()

#residual plot
plt.scatter(lm_predict_train, lm_predict_train - Y_train, c = 'b', s = 40, alpha = 0.5)
plt.scatter(lm_predict_test, lm_predict_test - Y_test, c = 'g', s =40)
plt.hlines(y = 0, xmin = 0, xmax = 50)
plt.title('Residual Plot using Train(blue) and Test(green)')
plt.ylabel('residual')

#2nd model: regression use scaled numeric features
X_train_scale=scale(X_train[NumFeatures])
X_test_scale=scale(X_test[NumFeatures])

scaledlm = linear_model.LinearRegression()
scaledlm.fit(X_train_scale, Y_train)
scaledlm_predict_train = scaledlm.predict(X_train_scale)
scaledlm_predict_test = scaledlm.predict(X_test_scale)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, scaledlm_predict_train)))
#Root mean squared error for train 896.74
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, scaledlm_predict_test)))
#Root mean squared error for test: 893.58


#3rd model, include categorical features with LabelEncoder
le=LabelEncoder()
#Cat_Train = X_train.select_dtypes(include=[object])
Cat_Train = X_train[CatFeatures]
Cat_Train = Cat_Train.apply(le.fit_transform)
#Cat_Train.head()
enc = OneHotEncoder()
enc.fit(Cat_Train)
onehotlabels_train = enc.transform(Cat_Train).toarray()
#onehotlabels_train.shape

Cat_test = X_test[CatFeatures]
Cat_test = Cat_test.apply(le.fit_transform)
#Cat_test.head()
enc = OneHotEncoder()
enc.fit(Cat_test)
onehotlabels_test = enc.transform(Cat_test).toarray()
#onehotlabels_test.shape

catlm = linear_model.LinearRegression()
catlm.fit(onehotlabels_train, Y_train)
catlm_predict_train = catlm.predict(onehotlabels_train)
catlm_predict_test = catlm.predict(onehotlabels_test)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, catlm_predict_train)))
#Root mean squared error for train 943.15
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, catlm_predict_test)))
#Root mean squared error for test: 941.34


#4th model, combine scaled numeric and labelencoded cat features
combined_train = pd.concat([pd.DataFrame(X_train_scale), pd.DataFrame(onehotlabels_train)], axis=1)
combined_test = pd.concat([pd.DataFrame(X_test_scale), pd.DataFrame(onehotlabels_test)], axis=1)

alllm = linear_model.LinearRegression()
alllm.fit(combined_train, Y_train)
alllm_predict_train = alllm.predict(combined_train)
alllm_predict_test = alllm.predict(combined_test)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, alllm_predict_train)))
#Root mean squared error for train 851.30
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, alllm_predict_test)))
#Root mean squared error for test: 850.90

#5th model, regression tree
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=0)
tree.fit(combined_train, Y_train)
tree_predict_train = tree.predict(combined_train)
tree_predict_test = tree.predict(combined_test)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, tree_predict_train)))
#Root mean squared error for train 2.99
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, tree_predict_test)))
#Root mean squared error for test:  1188.86

#6th model, random forest
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(max_depth=3, random_state=0)
RF.fit(combined_train, Y_train)
RF_predict_train = RF.predict(combined_train)
RF_predict_test = RF.predict(combined_test)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, RF_predict_train)))
#Root mean squared error for train 952.91
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, RF_predict_test)))
#Root mean squared error for test: 949.88

#7th model, GBM
from sklearn import ensemble
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GBM = ensemble.GradientBoostingRegressor(**params)
GBM.fit(combined_train, Y_train)
GBM_predict_train = GBM.predict(combined_train)
GBM_predict_test = GBM.predict(combined_test)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, GBM_predict_train)))
#Root mean squared error for train:  863.21
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, GBM_predict_test)))
#Root mean squared error for test: 867.61 

# #############################################################################
# Plot feature importance
feature_importance = GBM.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, combined_train.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


#8th model, Nearest Neighbor regression
from sklearn import neighbors
knn = neighbors.regression.KNeighborsRegressor(n_neighbors = 5)
knn.fit(combined_train, Y_train)
knn_predict_train = knn.predict(combined_train)
knn_predict_test = knn.predict(combined_test)
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, knn_predict_train)))
#Root mean squared error for train 244.37
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, knn_predict_test)))
#Root mean squared error for test: 244.91


#9th model, xgboost
from xgboost import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV

params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)
params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}


XGB = XGBRegressor(nthread=-1)
grid = RandomizedSearchCV(XGB, params, n_jobs=1)
grid.fit(combined_train, Y_train)
predict_train = grid.best_estimator_.predict(combined_train)
predict_test = grid.best_estimator_.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train 
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, predict_train)))
#Root mean squared error for train 265.11
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, predict_test)))
#Root mean squared error for test: 265.22


#10th model, SVR with rbf
from sklearn.svm import SVR
SVR = SVR(kernel='rbf', C=1e3, gamma=0.1)
SVR.fit(combined_train, Y_train)
predict_train = SVR.predict(combined_train)
predict_test = SVR.predict(combined_test)
print("Mean squared error for train: %.2f" % mean_squared_error(Y_train, predict_train))
#Mean squared error for train  59873.37
print("Mean squared error for test: %.2f" % mean_squared_error(Y_test, predict_test))
#Mean squared error for test: 59754.02
print("Root mean squared error for train: %.2f" % math.sqrt(mean_squared_error(Y_train, predict_train)))
#Root mean squared error for train 244.37
print("Root mean squared error for test: %.2f" % math.sqrt(mean_squared_error(Y_test, predict_test)))
#Root mean squared error for test: 244.91


















'''
External Weather API call: 
    WeatherStartLoc_StartTime, WeatherEndLoc_StartTime, 

Other ideas to consider: 
    - driver
        age, years of driving experience, years of driving experience in current city, avg driving speed-highway/local, driver ratings, #cars for this driver 
    - car
        age, make, accidents
    - passenger
        number of passengers, passenger gender/age, passenger tourist/local
    - city specific model
'''
Weather = pd.read_csv(r'C:\Users\ds1\OneDrive\Code\fenche\Trip_Prediction\weather_data.csv')
Weather.head()
Weather.Date.min() #42005
Weather.Date.max() #42370

from datetime import timedelta, date
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

start_dt = date(2015, 1, 1)
end_dt = date(2016, 1, 1)
oneyeardate = []
for dt in daterange(start_dt, end_dt):
    oneyeardate = oneyeardate + [dt.strftime("%Y-%m-%d")]
        
dateDF = pd.DataFrame({'start_date':oneyeardate})   

weatherDF = pd.concat([Weather, dateDF], axis=1)
weatherDF.info()
weatherDF.weather_date.head()

#for train
Train.shape #(12905715, 27)
Train.info()
Train.head()
Train.start_date.head()
Train.start_date.min() #datetime.date(2015, 1, 1)
Train.start_date.max() #datetime.date(2016, 1, 1)

TrainNew = pd.merge(Train, weatherDF, on='start_date', how='outer')
TrainNew = pd.merge(left=Train,right=weatherDF, how='left',  left_on='start_date', right_on='weather_date')
TrainNew = pd.merge(Train, weatherDF, all.x = TRUE, by=c('start_date','weather_date'))
TrainNew = pd.merge(left=Train,right=weatherDF, how='outer',  left_on='start_date', right_on='weather_date')

TrainNew = Train.join(weatherDF.set_index('start_date'), on='start_date')
TrainNew = Train.merge(weatherDF,how='left', left_on='start_date', right_on='start_date')
TrainNew.Snow_Depth.value_counts


TrainNew.isnull().values.any()
len(TrainNew.isnull())
TrainNew.info()
TrainNew.head()
TrainNew.shape

TrainNew = pd.concat([Train, weatherDF],  axis=1, keys = ['start_date'])
TrainNew.info()
TrainNew.head()
TrainNew.shape



'''

Models to consider:
    - linear model + Regularization
    - GAM
    - Random Forest
    - XGBoost
    - Neural Nets

Special Model:
    - convert continuous Y to multi-class labels
    - build classification model
    - when scoring, provide predicted bucket, mean seconds in each bucket 
    
Cross-Validation:
    -Mean Absolute Prediction Error

Model-persistence
    - pickle model
    - save model
    - call model
    
Data Exploration Charts
    - Univariate x-y
    - Heat Map for longitude/latitude
    - 
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#Train.to_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv',index=False)
#Test.to_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\test_clean.csv',index=False)

Train = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\train_clean.csv')
Test = pd.read_csv(r'C:\Users\ds1\Downloads\HomeWork_Lyft\test_clean.csv')
#Train.isnull().values.any()
#Train.head()
Train.shape #(12905715, 27)
Test.shape #(  1434344, 26)

import sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder, scale, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import math 

#some correlation plots
plt.scatter(Train['haversineDist'], Train['duration'])
plt.xlabel('haversineDist')
plt.ylabel('duration')
plt.title('haversineDist vs duration')
plt.show()

plt.scatter(Train['DistanceStartToCityCenter'], Train['duration'])
plt.xlabel('DistanceStartToCityCenter')
plt.ylabel('duration')
plt.title('DistanceStartToCityCenter vs duration')
plt.show()

plt.scatter(Train['HeadingToCityCenter'], Train['duration'])
plt.xlabel('HeadingToCityCenter')
plt.ylabel('duration')
plt.title('HeadingToCityCenter vs duration')
plt.show()

plt.scatter(Train['geoDist'], Train['duration'])
plt.xlabel('geoDist')
plt.ylabel('duration')
plt.title('geoDist vs duration')
plt.show()
   
plt.scatter(Train['start_hour'], Train['duration'])
plt.xlabel('start_hour')
plt.ylabel('duration')
plt.title('start_hour vs duration')
plt.show()
   
plt.scatter(Train['start_isWeekend'], Train['duration'])
plt.xlabel('start_isWeekend')
plt.ylabel('duration')
plt.title('start_isWeekend vs duration')
plt.show()
   
plt.scatter(Train['start_isHoliday'], Train['duration'])
plt.xlabel('start_isHoliday')
plt.ylabel('duration')
plt.title('start_isHoliday vs duration')
plt.show()


durationVSstart_dayOfWeek = removedExtremeDurations['duration'].groupby(removedExtremeDurations['start_dayOfWeek']).mean().to_frame()
plt.plot(durationVSstart_dayOfWeek)

durationVSstart_hour = removedExtremeDurations['duration'].groupby(removedExtremeDurations['start_hour']).mean().to_frame()
plt.plot(durationVSstart_hour)

durationVSstart_hour = removedExtremeDurations['duration'].groupby(removedExtremeDurations['start_hour']).mean().to_frame()
plt.plot(durationVSstart_hour)








