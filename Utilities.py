def create_summary_table(df):
    
    def num_missing(x):
        return len(x.index)-x.count()
    
    def pct_missing(x):
        return 100*(len(x.index)-x.count())/x.count()

    def num_unique(x):
        return len(np.unique([a for a in x if a is not None]))

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
    col_names = list(Train.columns)
    num_cols = len(col_names)
    index = range(num_cols)
    cat_index = []

    for i in index:
        if Train.dtypes[i] == 'object':
            cat_index.append(i)

    summary_df_cat = missing_df.join(unq_df).join(types_df.iloc[cat_index], how='inner')
    summary_df_cat

    return summary_df_cont, summary_df_cat