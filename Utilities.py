import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt

# create summary table for numerical and categorical features
def create_summary_table(df):
    def num_missing(x):
        return len(x.index)-x.count()  
    def pct_missing(x):
        return (len(x.index)-x.count())/x.count()
    def num_unique(x):
        return len(x.value_counts(dropna=False))

    # for numerical features
    temp_df = df.describe().T
    missing_df = pd.DataFrame(df.apply(num_missing, axis=0)) 
    missing_df.columns = ['missing']
    pct_missing_df = pd.DataFrame(df.apply(pct_missing, axis=0))
    pct_missing_df.columns = ['pct_missing']
    unq_df = pd.DataFrame(df.apply(num_unique, axis=0))
    unq_df.columns = ['unique']
    types_df = pd.DataFrame(df.dtypes)
    types_df.columns = ['DataType']   
    summary_df_cont = temp_df.join(missing_df).join(pct_missing_df).join(unq_df).join(types_df)
    
    # for Cat features
    col_names = list(df.columns)
    num_cols = len(col_names)
    index = range(num_cols)
    cat_index = []
    for i in index:
        if df.dtypes[i] == 'object':
            cat_index.append(i)
    summary_df_cat = missing_df.join(unq_df).join(types_df.iloc[cat_index], how='inner')
    summary_df_cat
    return summary_df_cont, summary_df_cat


# create target variable
def create_target(df):
    # drop those records that have missing actual delivery time
    df = df[pd.notnull(df['actual_delivery_time'])]
    df['created_at_datetime'] = df['created_at'].astype("datetime64[s]")
    df['actual_delivery_time_datetime'] = df['actual_delivery_time'].astype("datetime64[s]")
    df['duration'] = df['actual_delivery_time_datetime'] - df['created_at_datetime']
    df['duration'] = df['duration'] / np.timedelta64(1, 's')
    return df

# create features based on the timing of order creation time
def create_time_feature(df):
    #create created_at_year, created_at_month, created_at_day, created_at_date, created_at_dayOfWeek, 
    #created_at_time, created_at_hour, created_at_minute, created_at_second, created_at_isWeekend,
    #created_at_isHoliday
    df['created_at_year'], df['created_at_month'], df['created_at_day'], df['created_at_date'], df['created_at_dayOfWeek'], df['created_at_time'], df['created_at_hour'], df['created_at_minute'], df['created_at_second'] = df['created_at_datetime'].dt.year, df['created_at_datetime'].dt.month, df['created_at_datetime'].dt.day, df['created_at_datetime'].dt.date, df['created_at_datetime'].dt.dayofweek, df['created_at_datetime'].dt.time, df['created_at_datetime'].dt.hour, df['created_at_datetime'].dt.minute, df['created_at_datetime'].dt.second
    df.loc[df['created_at_dayOfWeek'].isin([5, 6]), 'created_at_isWeekend'] = 1
    df.loc[df['created_at_dayOfWeek'].isin([0, 1, 2, 3, 4]), 'created_at_isWeekend'] = 0
    cal = calendar()
    holidays = cal.holidays(start=df['created_at_date'].min(), end=df['created_at_date'].max())
    df['created_at_isHoliday'] = np.where(df.created_at_datetime.dt.normalize().isin(holidays), 1, 0)
    return df

# create plot for target variable vesus features
def create_plot(df, feature, target):
    plt.bar(df[feature], df[target], align='center', alpha=0.5)
    plt.xticks(df[feature], df[feature].values)
    plt.ylabel('Average ' + str(target))
    plt.xlabel(str(feature))
    plt.title('Average ' + str(target) + ' Vs ' + str(feature))
    plt.show()