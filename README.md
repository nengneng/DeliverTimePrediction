# DeliverTimePrediction

## Data Processing:
1. Calcualte True Label 
    - time diff
2. Time Feature:
    - Day, Hour, Weekend, IsHoliday, 
3. Weather feature
    - Download
    - Merge
4. Categorical Features 
    - encoding
    - re-grouping
5. Continuous Featurs
    - direct use
    - binning    
6. Missing Values
    - Categorical: UNK group
    - Continuous: imputing (mean, medain, mode)
7. Store_Id
    - Hashing

## Models:
1. Linear Model
2. RF
3. GBM
4. XGBoost
5. NN
6. DNN

## Ideas:
- If geo location is explicit, we may have traffic info
- Driving Speed
- Car Info
- Driver Info
- Highway/Local
- Speed info
- Ratings of customers
- Ratings of Dashers

## Application Infrastructure
- Save pickled best model
- Write Scoring function
- Utilities

