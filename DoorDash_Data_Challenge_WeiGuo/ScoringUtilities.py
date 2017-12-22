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

def process_continuous_features(df):
    
    def bin_num(x, a=251, b=446):
        if x == a:
             return 'fast'
        elif x == b:
             return 'slow'
        else:
            return 'other'
    
    df['total_items'][(df['total_items'] > 20)] = 20
    df['subtotal'][df['subtotal'] > 12000] = 12000
    df['num_distinct_items'][df['num_distinct_items'] > 16] = 16
    df['min_item_price'][(df['min_item_price'] < 0)] = 0
    df['min_item_price'][(df['min_item_price'] > 5000)] = 5000

    df['max_item_price'][(df['max_item_price'] < 0)] = 0
    df['max_item_price'][(df['max_item_price'] > 5000)] = 5000

    df['total_onshift_dashers'][df['total_onshift_dashers'] < 0] = 0
    df['total_onshift_dashers'] = df['total_onshift_dashers'].fillna(int(df['total_onshift_dashers'].mean()))
    
    df['total_busy_dashers'][df['total_busy_dashers'] < 0] = 0
    df['total_busy_dashers'] = df['total_busy_dashers'].fillna(int(df['total_busy_dashers'].mean()))
    
    df['total_outstanding_orders'][df['total_outstanding_orders'] < 0] = 0
    df['total_outstanding_orders'] = df['total_outstanding_orders'].fillna(int(df['total_outstanding_orders'].mean()))
    
    df['estimated_order_place_duration_rebinned'] =  df['estimated_order_place_duration'].apply(bin_num)
    df['estimated_store_to_consumer_driving_duration'] = df['estimated_store_to_consumer_driving_duration'].fillna(int(df['estimated_store_to_consumer_driving_duration'].mean()))
    return df

def make_store_id_cont_score(df):
    df['store_id_count'][df['store_id_count'].isnull()] = 0   
    df['store_id_rebinned'] = df['store_id']
    df['store_id_rebinned'][(df['store_id_count'] <500) & (df['store_id_count'] >= 400)] = '[400, 500)'
    df['store_id_rebinned'][(df['store_id_count'] <400) & (df['store_id_count'] >= 200)] = '[200, 400)'
    df['store_id_rebinned'][(df['store_id_count'] <200) & (df['store_id_count'] >= 50)] = '[50, 200)'
    df['store_id_rebinned'][(df['store_id_count'] <50)] = '[0, 50)'
    return df

def make_store_id_cont(df):
    store_counts_df = pd.DataFrame(df['store_id'].value_counts().reset_index().rename(columns={'index': 'store_id', 0: 'store_id_count'}))
    store_counts_df.columns = ['store_id', 'store_id_count']
    store_counts_df = store_counts_df.sort_values(by='store_id', ascending=True)
    df = pd.merge(df, store_counts_df, on='store_id', how='left')
    df['store_id_rebinned'] = df['store_id']
    df['store_id_rebinned'][(df['store_id_count'] <500) & (df['store_id_count'] >= 400)] = '[400, 500)'
    df['store_id_rebinned'][(df['store_id_count'] <400) & (df['store_id_count'] >= 200)] = '[200, 400)'
    df['store_id_rebinned'][(df['store_id_count'] <200) & (df['store_id_count'] >= 50)] = '[50, 200)'
    df['store_id_rebinned'][df['store_id_count'] <50] = '[0, 50)'
    return df,store_counts_df

def make_store_category_cont_score(df):
    df['store_primary_category_count'][df['store_primary_category_count'].isnull()] = 0    
    df['store_primary_category_rebinned'] = df['store_primary_category']
    df['store_primary_category_rebinned'][df['store_primary_category_rebinned'].isnull()] = 'Unknown'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <3000) & (df['store_primary_category_count'] >= 2000)] = '[2000, 3000)'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <2000) & (df['store_primary_category_count'] >= 1000)] = '[1000, 2000)'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <1000) & (df['store_primary_category_count'] >= 200)] = '[200, 1000)'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <200)] = '[0, 200)'
    return df

def make_store_category_cont(df):
    df['store_primary_category'][df['store_primary_category'].isnull()] = 'Unknown'   
    store_primary_category_counts_df = pd.DataFrame(df['store_primary_category'].value_counts().reset_index().rename(columns={'index': 'store_primary_category', 0: 'store_primary_category_count'}))
    store_primary_category_counts_df.columns = ['store_primary_category', 'store_primary_category_count']
    df = pd.merge(df, store_primary_category_counts_df, on='store_primary_category', how='left')
    df['store_primary_category_rebinned'] = df['store_primary_category']
    df['store_primary_category_rebinned'][df['store_primary_category_rebinned'].isnull()] = 'Unknown'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <3000) & (df['store_primary_category_count'] >= 2000)] = '[2000, 3000)'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <2000) & (df['store_primary_category_count'] >= 1000)] = '[1000, 2000)'
    df['store_primary_category_rebinned'][(df['store_primary_category_count'] <1000) & (df['store_primary_category_count'] >= 200)] = '[200, 1000)'
    df['store_primary_category_rebinned'][df['store_primary_category_count'] <200] = '[0, 200)'
    return df, store_primary_category_counts_df

def impute_market_id(df):
    df['market_id'][df['market_id'].isnull()] = 0
    return df

def impute_order_protocol(df):
    df['order_protocol'][df['order_protocol'].isnull()] = 0
    df['order_protocol'].loc[df['order_protocol'] == 6] = 0
    df['order_protocol'].loc[df['order_protocol'] == 7] = 0
    return df

def select_features(df,TrainOrScore):
    if TrainOrScore == 'Train':
        TrainFeatures = df[['duration', 'market_id', 'store_id_rebinned', 'store_primary_category_rebinned',
                            'order_protocol',  'total_items', 'subtotal', 'num_distinct_items', 'min_item_price',
                            'max_item_price','total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders',
                            'estimated_store_to_consumer_driving_duration', 'created_at_month', 'created_at_dayOfWeek',
                            'created_at_hour', 'created_at_isWeekend', 'created_at_isHoliday',
                            'estimated_order_place_duration_rebinned']]
    else:
        TrainFeatures = df[['market_id', 'store_id_rebinned', 'store_primary_category_rebinned',
                    'order_protocol',  'total_items', 'subtotal', 'num_distinct_items', 'min_item_price',
                    'max_item_price','total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders',
                    'estimated_store_to_consumer_driving_duration', 'created_at_month', 'created_at_dayOfWeek',
                    'created_at_hour', 'created_at_isWeekend', 'created_at_isHoliday',
                    'estimated_order_place_duration_rebinned']]
        
    TrainFeatures[['market_id', 'store_id_rebinned', 'store_primary_category_rebinned', 'order_protocol', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour',  'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']] = TrainFeatures[['market_id', 'store_id_rebinned', 'store_primary_category_rebinned', 'order_protocol', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour',  'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']].astype(object)
    NumFeatures = ['total_items', 'subtotal', 'num_distinct_items', 'min_item_price',  'max_item_price','total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_store_to_consumer_driving_duration']
    CatFeatures = ['market_id', 'store_id_rebinned', 'store_primary_category_rebinned',  'order_protocol', 'created_at_month', 'created_at_dayOfWeek', 'created_at_hour', 'created_at_isWeekend', 'created_at_isHoliday', 'estimated_order_place_duration_rebinned']
    return TrainFeatures, NumFeatures, CatFeatures

def scale_oneHot_X(input_x, features_num, features_cat):
    # scale numerical features
    input_x_scale = scale(input_x[features_num])
    
    # OneHot cat features
    le=LabelEncoder()
    enc = OneHotEncoder()
    Cat_Train = input_x[features_cat].apply(le.fit_transform)
    enc.fit(Cat_Train)
    input_x_oneHot = enc.transform(Cat_Train).toarray()
    output_x = pd.concat([pd.DataFrame(input_x_scale), pd.DataFrame(input_x_oneHot)], axis=1)
    output_x.columns = [i for i in range(output_x.shape[1])]
    return output_x 

def make_prediction(model, data):
    model_loaded = joblib.load(model)
    pred = model_loaded.predict(data)
    return pred

def load_unlabeled_data(input_file):
    loaded_data = []
    with open(input_file) as f:
        for line in f:
            loaded_data.append(json.loads(line))
            
    created_at_lst = [x['created_at'] for x in loaded_data]
    delivery_id_lst = [x['delivery_id'] for x in loaded_data]
    estimated_order_place_duration_lst = [x['estimated_order_place_duration'] for x in loaded_data]
    estimated_store_to_consumer_driving_duration_lst = [x['estimated_store_to_consumer_driving_duration'] for x in loaded_data]
    market_id_lst = [x['market_id'] for x in loaded_data]
    max_item_price_lst = [x['max_item_price'] for x in loaded_data]
    min_item_price_lst = [x['min_item_price'] for x in loaded_data]
    num_distinct_items_lst = [x['num_distinct_items'] for x in loaded_data]
    order_protocol_lst = [x['order_protocol'] for x in loaded_data]
    platform_lst = [x['platform'] for x in loaded_data]
    store_id_lst = [x['store_id'] for x in loaded_data]
    store_primary_category_lst = [x['store_primary_category'] for x in loaded_data]
    subtotal_lst = [x['subtotal'] for x in loaded_data]
    total_busy_dashers_lst = [x['total_busy_dashers'] for x in loaded_data]
    total_items_lst = [x['total_items'] for x in loaded_data]
    total_onshift_dashers_lst = [x['total_onshift_dashers'] for x in loaded_data]
    total_outstanding_orders_lst = [x['total_outstanding_orders'] for x in loaded_data]
    
    unlabled_df = pd.DataFrame(
        {'created_at': created_at_lst,
         'delivery_id': delivery_id_lst,
         'estimated_order_place_duration': estimated_order_place_duration_lst,
         'estimated_store_to_consumer_driving_duration': estimated_store_to_consumer_driving_duration_lst,
         'market_id': market_id_lst,
         'max_item_price': max_item_price_lst,
         'min_item_price': min_item_price_lst,
         'num_distinct_items': num_distinct_items_lst,
         'order_protocol': order_protocol_lst,
         'platform': platform_lst,
         'store_id': store_id_lst,
         'store_primary_category': store_primary_category_lst,
         'subtotal': subtotal_lst,
         'total_busy_dashers': total_busy_dashers_lst,
         'total_items': total_items_lst,
         'total_onshift_dashers': total_onshift_dashers_lst,
         'total_outstanding_orders': total_outstanding_orders_lst
        })
    return unlabled_df

def create_target(df):
    # drop those records that have missing actual delivery time
    df = df[pd.notnull(df['actual_delivery_time'])]
    df['duration'] = df['actual_delivery_time'].astype("datetime64[s]") - df['created_at'].astype("datetime64[s]")
    df['duration'] = df['duration'] / np.timedelta64(1, 's')
    return df
