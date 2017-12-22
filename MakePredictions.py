import pandas as pd
import ScoringUtilities
import sys

store_category_count_table_df = pd.read_csv(store_category_count_table_file)
make_store_id_cont_table_df = pd.read_csv(make_store_id_cont_table_file)
Train = pd.read_csv(train_file)

unlabeled_df = ScoringUtilities.load_unlabeled_data(unlabeled_json)
unlabeled_df_merge_1 = unlabeled_df.merge(store_category_count_table_df, left_on='store_primary_category', right_on='store_primary_category', how='left')
unlabeled_df_merge_2 = unlabeled_df_merge_1.merge(make_store_id_cont_table_df, left_on='store_id', right_on='store_id', how='left')
unlabeled_df_merge_2[['market_id','estimated_order_place_duration','estimated_store_to_consumer_driving_duration',
             'max_item_price','min_item_price', 'num_distinct_items', 'order_protocol',
             'subtotal','total_onshift_dashers','total_busy_dashers','total_items','total_onshift_dashers',
             'total_outstanding_orders', 'store_primary_category_count', 'store_id_count']] = unlabeled_df_merge_2[['market_id','estimated_order_place_duration','estimated_store_to_consumer_driving_duration',
             'max_item_price','min_item_price', 'num_distinct_items', 'order_protocol',
             'subtotal','total_onshift_dashers','total_busy_dashers','total_items','total_onshift_dashers',
             'total_outstanding_orders', 'store_primary_category_count', 'store_id_count']].apply(pd.to_numeric, errors='coerce')

def process_unlabel(df):
    a1 = ScoringUtilities.create_time_feature(df)
    a = ScoringUtilities.process_continuous_features(a1)
    b = ScoringUtilities.impute_market_id(a)
    c = ScoringUtilities.impute_order_protocol(b)
    d = ScoringUtilities.make_store_category_cont_score(c)
    e = ScoringUtilities.make_store_id_cont_score(d)
    unlabeled_ready = ScoringUtilities.select_features(e, 'Test')[0]
    return unlabeled_ready, e

def process_train(df):
    a0 = ScoringUtilities.create_target(df)
    a1 = ScoringUtilities.create_time_feature(a0)
    a = ScoringUtilities.process_continuous_features(a1)
    b = ScoringUtilities.impute_market_id(a)
    c = ScoringUtilities.impute_order_protocol(b)
    d = ScoringUtilities.make_store_category_cont(c)[0]
    e = ScoringUtilities.make_store_id_cont(d)[0]
    Train_ready = ScoringUtilities.select_features(e, 'Train')[0]
    del Train_ready['duration']
    return Train_ready

unlabeled_ready = process_unlabel(unlabeled_df_merge_2)[0]
Train_ready = process_train(Train)
train_plus_unlabel = pd.concat([Train_ready, unlabeled_ready], axis=0)
NumFeatures = ScoringUtilities.select_features(process_unlabel(unlabeled_df_merge_2)[1], 'Test')[1]
CatFeatures = ScoringUtilities.select_features(process_unlabel(unlabeled_df_merge_2)[1], 'Test')[2]
train_plus_unlabel_encoded = ScoringUtilities.scale_oneHot_X(train_plus_unlabel, NumFeatures,CatFeatures)
unlabeled_ready2 = train_plus_unlabel_encoded.tail(unlabeled_ready.shape[0])

gbm_pred = ScoringUtilities.make_prediction(model_name, unlabeled_ready2)

print("{} scored {} records.".format(model_name, len(gbm_pred)))
print("First 20 records are: \n {} ".format(gbm_pred[:20]))

id = unlabeled_df['delivery_id'].tolist()
output_df = pd.DataFrame({'delivery_id': unlabeled_df['delivery_id'], 'predicted_delivery_seconds': gbm_pred})
output_df.to_csv(output_file, index=False, sep="\t")

if __name__ == "__main__":
    unlabeled_json = sys.argv[1]
    store_category_count_table_file = sys.argv[2]
    make_store_id_cont_table_file = sys.argv[3]
    train_file = sys.argv[4]
    model_name = sys.argv[5]
    output_file = sys.argv[6]

print("Scored data is saved in {}".format(output_file))