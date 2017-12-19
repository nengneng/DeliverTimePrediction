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
Train2 = Train[pd.notnull(Train['actual_delivery_time'])]

#conver time object to datetime64
Train2['created_at_datetime'] = Train2['created_at'].astype("datetime64[s]")
Train2['actual_delivery_time_datetime'] = Train2['actual_delivery_time'].astype("datetime64[s]")

#create label: duration
Train2['duration'] = Train2['actual_delivery_time_datetime'] - Train2['created_at_datetime']
Train2['duration'] = Train2['duration'] / np.timedelta64(1, 's')

#create created_at_year, created_at_month, created_at_day, created_at_date, created_at_dayOfWeek, 
#created_at_time, created_at_hour, created_at_minute, created_at_second, created_at_isWeekend,
#created_at_isHoliday 
Train2['created_at_year'], Train2['created_at_month'], Train2['created_at_day'], Train2['created_at_date'], Train2['created_at_dayOfWeek'], Train2['created_at_time'], Train2['created_at_hour'], Train2['created_at_minute'],  Train2['created_at_second'] = Train2['created_at_datetime'].dt.year, Train2['created_at_datetime'].dt.month, Train2['created_at_datetime'].dt.day, Train2['created_at_datetime'].dt.date, Train2['created_at_datetime'].dt.dayofweek, Train2['created_at_datetime'].dt.time, Train2['created_at_datetime'].dt.hour, Train2['created_at_datetime'].dt.minute, Train2['created_at_datetime'].dt.second

Train2.loc[Train2['created_at_dayOfWeek'].isin([5, 6]), 'created_at_isWeekend'] = 1
Train2.loc[Train2['created_at_dayOfWeek'].isin([0, 1, 2, 3, 4]), 'created_at_isWeekend'] = 0

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
cal = calendar()
holidays = cal.holidays(start=Train2['created_at_date'].min(), end=Train2['created_at_date'].max())
Train2['created_at_isHoliday'] = np.where(Train2.created_at_datetime.dt.normalize().isin(holidays), 1, 0)

#time features description

#daily counts of 
TimeSeriesTrain = pd.DataFrame(Train2['created_at_date'].value_counts().reset_index().rename(columns={'index': 'start_date', 0: 'Count'}))
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

Train2['created_at_year'].value_counts()/Train2['created_at_year'].value_counts().sum()
#2015    197420
#2014         1

Train2['created_at_date'].value_counts()/Train2['created_at_date'].value_counts().sum()
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

Train2['created_at_month'].value_counts()/Train2['created_at_month'].value_counts().sum()
#2     128938
#1      68482
#10         1

Train2['created_at_dayOfWeek'].value_counts()/Train2['created_at_dayOfWeek'].value_counts().sum()
#5 and 6 are weekends, which have more orders
#5    0.174961
#6    0.170271
#4    0.141196
#0    0.138800
#3    0.130042
#2    0.122854
#1    0.121877

Train2['created_at_isWeekend'].value_counts()/Train2['created_at_isWeekend'].value_counts().sum()
#0.0    0.654768
#1.0    0.345232

Train2['created_at_hour'].value_counts()/Train2['created_at_hour'].value_counts().sum()
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
Duration description
'''

Train2.duration.describe()
#maybe remove outlier: max value
#count    1.974210e+05
#mean     2.908257e+03
#std      1.922961e+04
#min      1.010000e+02
#25%      2.104000e+03
#50%      2.660000e+03
#75%      3.381000e+03
#max      8.516859e+06

plt.hist(Train2.duration, normed=True, bins=50)
plt.hist(Train2.duration)

Train3 = Train2.loc[Train2['duration'] < 8516859]
len(Train3)/len(Train2) #0.9999949346827338  #remove the max value
plt.hist(Train3.duration, normed=True, bins=50)


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
Train3['market_id'].value_counts()/Train3['market_id'].value_counts().sum()
#2.0    0.280274
#4.0    0.242307
#1.0    0.193633
#3.0    0.118595
#5.0    0.091629
#6.0    0.073562

Train3['store_id'].value_counts()/Train3['store_id'].value_counts().sum()
#very imbalanced, consider binning 

store_counts = Train3['store_id'].value_counts()
Train3[Train3['store_id'].isin(store_counts[store_counts == 1].index)]['store_id']
len(Train3[Train3['store_id'].isin(store_counts[store_counts == 1].index)]) / len(Train3)
#0.0028669840948232195 #566 stores have only 1 order

Train3[Train3['store_id'].isin(store_counts[store_counts == 2].index)]['store_id']
len(Train3[Train3['store_id'].isin(store_counts[store_counts == 2].index)]) / len(Train3) 
#0.004832337149225003 #954 store have only 2 orders

Train3[Train3['store_id'].isin(store_counts[store_counts == 3].index)]['store_id']
len(Train3[Train3['store_id'].isin(store_counts[store_counts == 3].index)]) / len(Train3) 
#0.006123999594772566 #1209 stores have only 3 orders

store_counts_df = pd.DataFrame(Train3['store_id'].value_counts().reset_index().rename(columns={'index': 'store_id', 0: 'Count'}))
store_counts_df.columns = ['store_id', 'Count']
store_counts_df = store_counts_df.sort_values(by='store_id', ascending=True)

plt.figure(figsize=(30,10))
plt.rcParams.update({'font.size': 22})
plt.plot(store_counts_df['store_id'], store_counts_df['Count'])
plt.tight_layout()

store_counts_df['Count'].describe()



