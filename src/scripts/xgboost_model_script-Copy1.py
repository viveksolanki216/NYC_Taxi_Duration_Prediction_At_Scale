import os
import pandas as pd
import argparse
import xgboost as xgb
import pickle
import numpy as np
import json
#from sklearn.metrics import root_mean_squared_error
import gc
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack, csr_matrix, hstack


def get_parameters():
    parser = argparse.ArgumentParser(description="Train a model for fraud detection")
    parser.add_argument("--train-dir",       type=str, default=os.environ.get("SM_CHANNEL_TRAIN"), help="Directory containing the training data")
    parser.add_argument("--test-dir",        type=str, default=os.environ.get("SM_CHANNEL_TEST"), help="Directory containing the +testing data")
    parser.add_argument("--model-out-dir",         type=str, default=os.environ.get("SM_MODEL_DIR"), help="Directory to save the trained model")
    parser.add_argument("--model-data-out-dir",    type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"), help="Directory to save the trained model")
    parser.add_argument("--target-var",            type=str, default='fraud', help="Target variable for the model", required=True)
    parser.add_argument("--features",              type=str, default=None, help="| list of feature variables", required=True)
    parser.add_argument("--max-depth",             type=int, default=6, help="Maximum depth of the tree", required=True)
    parser.add_argument("--eta",                   type=float, default=0.3, help="Learning rate", required=True)
    parser.add_argument("--objective",             type=str, help="Learning task and objective function.", required=True)
    parser.add_argument("--num-boost-round",       type=int, default=1, help="Number of boosting rounds", required=True)
   
    print("Parsing arguments...")
    args = parser.parse_args()
    print(args)

    hyper_params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective
    }
    return args.train_dir, args.test_dir, args.model_out_dir, args.model_data_out_dir, args.target_var, args.features, args.num_boost_round, hyper_params

# Prints the available elements only of a CSR matrix
def print_csr_matrix(X):
    for i in range(X.shape[0]):
        start = X.indptr[i]
        end = X.indptr[i+1]
        cols = X.indices[start:end]
        vals = X.data[start:end]
        print(f"Row {i}: {list(zip(cols, vals))}")

# converts the sparse dictionary to csr matrix
def sparse_dict_to_csr(row):
    size = row['size']
    indices = row['indices']
    values = row['values']
    # row vector, shape=(1, size)
    return csr_matrix((values, (np.zeros_like(indices), indices)), shape=(1, size))

def get_scipy_sparse_csr_matrix(data, numeric_columns, onehot_vector_column='features'):
    '''
    Converts a pandas dataframe containing numerical columns and a single column that contains vector of one hot encoded dummy variables
    Return a scipy sparse csr matrix
    '''
    # First convert each value i.e. sparse dict to a sparse csr vertically stack them
    X_sparse = vstack([sparse_dict_to_csr(row) for row in data[onehot_vector_column]])
    print("Done1")
    # We need to convert all numeric features to sparse matrix too and horizontally stack with the above "features" sparse matrix 
    X_numeric = data[numeric_columns].to_numpy() # Dense matrix
    X_numeric_sparse = csr_matrix(X_numeric)   # Sparse matrix
    print("Done2")
    # concatenate the numeric and ohe columns into one numpy matrix
    X_all = hstack([X_numeric_sparse, X_sparse])
    print(X_all.shape)
    #print(X_all)
    #print_csr_matrix(X_all)
    return X_all
    
def read_csv_data(base_dir: str) -> pd.DataFrame:
    '''
    Read all csv files inside the directory
    '''
    print("READING FROM -", base_dir)
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            print(root)
            print(fname)
            if fname.endswith('.csv'):
                csv_files.append(os.path.join(root, fname))
   
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {base_dir}")

    print(csv_files)
    data = pd.concat(
        [pd.read_csv(fname) for fname in csv_files],
        ignore_index=True
    )
    print(f"data shape: {data.shape}")
    return data

    
if __name__ == "__main__":
    train_dir, test_dir, model_out_dir, model_data_out_dir, target_var, features, num_boost_round, hyper_params  = get_parameters()

    if features is not None:
        features = features.split('|')
    print(features)
    
    

    # READ TRAIN DATA
    #train = read_csv_data(train_dir)
    train = pd.read_parquet(train_dir)
    print("Train Data After Reading: ", train.shape)
    print(target_var)
    train_target = train[target_var]
    train = train[features]
    numeric_features = train.select_dtypes(include=["number"]).columns.tolist()
    print(numeric_features)
    
    # READ TEST DATA
    #test = read_csv_data(test_dir)
    test = pd.read_parquet(test_dir)
    print("Test Data After Reading: ", test.shape)
    test_target = test.pop(target_var)
    test = test[features]

    # CONVERT ONE-HOT-ENCODED VECTOR COLUMN 'FEATURES' AND OTHER NUMERIC FEATURES TO SCIPY CSR MATRIX
    # IF NO ONE-HOT-ENCODED THEN JUST USE DENSE MATRIX FOR NUMERIC FEATURES
    if 'features' in features: # if one hot encode feature vectors is saved under column "features"
        print("creating CSR matrix")
        train_np = get_scipy_sparse_csr_matrix(train, numeric_features, onehot_vector_column='features')
        print("Traing creating CSR matrix: Done")
        test_np = get_scipy_sparse_csr_matrix(test, numeric_features, onehot_vector_column='features')
        print("test creating CSR matrix: Done")
        
    else:
        train_np = train[numeric_features].values
        test_np = test[numeric_features].values

    # Delete train and test
    del train
    del test 
    gc.collect()

    print("Train and Test Sizes")
    print(train_np.shape, test_np.shape)
    #print(train_np.dtypes)
    
    # convert data to DMatrix
    dtrain = xgb.DMatrix(train_np, label=train_target)
    dtest = xgb.DMatrix(test_np, label=test_target)

    print("Train Size after converting to DMatrix")
    num_rows = dtrain.num_row()
    num_cols = dtrain.num_col()
    print(dtrain.num_row(), dtrain.num_col())

    # Delete train and test
    del train_np
    del test_np 
    gc.collect()

    
    # UPDATE THE HYPER PARAMS
    hyper_params['tree_method'] = 'hist'
    if hyper_params['objective'] == "reg:squarederror":
        hyper_params['eval_metric'] = 'rmse'

    # TRAINING
    # To store evaluation results at each iteration
    eval_results = {}
    bst = xgb.train(
        params=hyper_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=10,
        evals_result=eval_results
    )

    # STORE THE MODEL
    os.makedirs(model_out_dir, exist_ok=True)
    model_path = os.path.join(model_out_dir, )
    bst.save_model(f"{model_out_dir}/xgboost-model")
    #with open(model_location, "wb") as f:
    #    pickle.dump(model, f)

    # STORE THE TRAIN & TEST ACCURACY
    best_iteration = bst.best_iteration if bst.best_iteration is not None else len(eval_results["train"]["rmse"]) - 1

    train_rmse = eval_results["train"]["rmse"][best_iteration]
    test_rmse = eval_results["eval"]["rmse"][best_iteration]
    
    metrics_data = {
        "RMSE": {
            "train": train_rmse,
            "test": test_rmse
        }
    }

    print("model training done")
    # Save the model to the location specified by ``model_dir``
    metrics_location = model_data_out_dir + "/metrics.json"

    os.makedirs(model_data_out_dir, exist_ok=True)
    with open(metrics_location, "w") as f:
        json.dump(metrics_data, f)

