import json
import os
import argparse
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, round, concat_ws, lit, when,
    dayofweek,
    dayofmonth,
    month,
    hour,
    weekofyear
)
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

def get_parameters():
    '''
    Parses Command Line Arguments and returns value for them
    '''
    parser = argparse.ArgumentParser()
    #parser.add_argument("--s3file-path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['train', 'test'], default='train')
    parser.add_argument("--database-name", type=str, required=True)
    parser.add_argument("--table-name", type=str, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    return (
        args.mode, args.database_name, args.table_name, args.start_date, args.end_date,
        args.num_shards, args.out_path
    )

def get_data(spark, database_name: str, table_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    '''
    Extract data from the glue data-catalog table generated from schema corrected data using Glue ETL job
    '''
    query = f"""
        SELECT * FROM {database_name}.{table_name} 
        WHERE
            tpep_pickup_date >= '{start_date}' 
            AND tpep_pickup_date <= '{end_date}'
    """
    print(query)

    df = spark.sql(query)
    print("Read Data")

    # Can also read directly from S3 files. Just provide the directory as input.
    #df = spark.read.parquet(args.s3file_path)    
    
    return df


def process_and_transform_data(df):
    # TARGET
    df = df.withColumn("trip_duration_mins", round((col('tpep_dropoff_datetime').cast('long') - col('tpep_pickup_datetime').cast('long'))/60, 3))
    # FILTER OUT OUTLIERS & SELECT RELEVANT COLUMNS ONLY   
    df = df.filter(
        col('trip_duration_mins').between(1, 60) &
        col('trip_distance').between(0, 100) &
        col('passenger_count').between(0, 6)
    ).select(*[
        "trip_duration_mins",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "pulocationid",
        "dolocationid",
        "passenger_count",
        "trip_distance",
        "vendorid",
        "ratecodeid"
    ])
    # EXTRACT TIMELINE FEATURES FOR SEASONALITY
    df = (
        df
        .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))
        .withColumn("day_of_month", dayofmonth(col("tpep_pickup_datetime")))
        .withColumn("month_of_year", month(col("tpep_pickup_datetime")))
        .withColumn("hour_of_day", hour(col("tpep_pickup_datetime")))
        .withColumn("week_of_year", weekofyear("tpep_pickup_datetime")) 
    )
    # Combination of PICKUP-DROP location should be a strong feature. 
    df = df.withColumn("pick_drop_loc", concat_ws("-", col("pulocationid").cast("string"), col("dolocationid").cast("string")))
 
    target_var = "trip_duration_mins"
    numerical_cols = [
        "passenger_count", "trip_distance", "vendorid", "ratecodeid", "day_of_week", 
        "day_of_month", "month_of_year", "hour_of_day", "week_of_year"
    ]
    categorical_cols = ['pick_drop_loc']
 
    return df, target_var, numerical_cols, categorical_cols

def get_categorical_encoder_pipeline(categorical_columns):    
    # Need to encode strings to a number/indices
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep") # Takes a single input category columns and returns encoded indices
        for col in categorical_columns
    ]
    # The above indices will be used to create vectors of dummy variables
    ohe = OneHotEncoder(
        # can encode multiple columns at one time and returns a single vector column for each input column
        inputCols=[f"{col}_index" for col in categorical_columns],
        outputCols=[f"{col}_vec" for col in categorical_columns]
    )
    # Need vector assembler to create a single vector from different vectors of dummy variables for each categorical variable
    # Assemble all OHE vectors
    assembler = VectorAssembler(
        inputCols = [f"{c}_vec" for c in categorical_columns],
        outputCol = "features"
    )
    # combine/concate all steps into one pipeline i.e. first all columns indexers then ohe and then assembler
    pipeline = Pipeline(stages = indexers+[ohe, assembler])

    return pipeline



if __name__ == "__main__":

    local_in_dir = '/opt/ml/processing/input/' # mapped to a s3 location
    
    local_out_dir = '/opt/ml/processing/output/' # mapped to a s3 location
    os.makedirs(local_out_dir, exist_ok=True)

    # READ COMMAND LINE ARGUMENTS
    # ------------------------------------------------------------------------------------------------------------------
    mode, database_name, table_name, start_date, end_date, num_shards, out_path = get_parameters()

    # CREATE A SPARK SESSION, ENABLE HIVE SUPPORT FOR AWS GLUE DATA CATALOG TABLES 
    # ------------------------------------------------------------------------------------------------------------------
    spark = (
        SparkSession.builder 
        .appName("ProcessingNYCData") 
        .config("spark.sql.catalogImplementation", "hive") 
        .config("hive.metastore.client.factory.class", "com.amazonaws.glue.catalog.metastore.AWSGlueDataCatalogHiveClientFactory")
        #.config("spark.hadoop.fs.defaultFS", "file:///")
        .enableHiveSupport() 
        .getOrCreate()
    )
    print("spark session created")

    # READ DATA FROM TABLE
    # ------------------------------------------------------------------------------------------------------------------
    df = get_data(spark, database_name, table_name, start_date, end_date)
    # RUN IT ON A SAMPLE for script development and correction.
    # df = df.sample(withReplacement=False, fraction=0.0001, seed=42)

    # DATA PROCESSING
    # ------------------------------------------------------------------------------------------------------------------
    # Calculate target, extract features, combine features
    # These are stateless transformations row based/independent doesn't need full scan of data
    df, target, num_features, cat_features = process_and_transform_data(df)
 
    # ENCODE CATEGORICAL FEATURES
        # Encode it to indices/numbers using StringIndexer
        # create dummy factors from indices above using OneHotEncoder
        # All categorical factors will be converted to a single vector per row, vector contains 0/1 value for each category
        # These vectors (each vector for a row) will be stored as single column with column name ="features"    
    # Stateful transformations, need scan of whole data, test data will need train data attributes
    if mode == "train":
        # first convert the rare categories of pickup_drop_loc to 'Other'
        threshold = 20
        common_categories_list = []
        for cat_f in cat_features:
        # Find common categories
            category_counts = df.groupBy(cat_f).count()
            common_categories = category_counts.filter(col("count") >= threshold) \
                                               .select(cat_f) \
                                               .rdd.flatMap(lambda x: x) \
                                               .collect()
            common_categories_list.append(common_categories)
        
            # Save to JSON file
            with open(f"{local_out_dir}/common_categories_{cat_f}.json", "w") as f:
                json.dump(common_categories, f)

        # One Hot Encoding
        pipeline = get_categorical_encoder_pipeline(cat_features)
        model = pipeline.fit(df)
        # Save the fitted pipeline    
        #local_out_dir = '/opt/ml/processing/output/preprocessor_model' # mapped to a s3 location
        #os.makedirs(local_out_dir, exist_ok=True)
        model.save(f"{out_path}/preprocessor_model")
        print(f"Pipeline model saved to ", f"{out_path}/preprocessor_model")
        
    elif mode == "test":
        # Load saved Common categories
        common_categories_list = []
        for cat_f in cat_features:
            with open(f"{local_in_dir}/common_categories_{cat_f}.json", "r") as f:
                common_categories = json.load(f)
                common_categories_list.append(common_categories)
        
        # Load the fitted pipeline
        model = PipelineModel.load(f"{out_path}/preprocessor_model")
        print(f"Pipeline model loaded from ", f"{out_path}/preprocessor_model")
    
    else:
        raise ValueError("Unknown mode! Use 'train' or 'test'.")

    # Assign Rare categories to Other Bin
    for cat_f, common_categories in zip(cat_features, common_categories_list): 
        df = df.withColumn(
            cat_f,
            when(col(cat_f).isin(common_categories), col(cat_f)).otherwise(lit("Other"))
        )
        
    # One Hot Encoder Transformation
    transformed_df = model.transform(df)
    
    
    cols_to_select = [target] + ["tpep_pickup_datetime", "tpep_dropoff_datetime", "pick_drop_loc"] + num_features + ["features"]
    transformed_df = transformed_df.select(*cols_to_select)
    print(transformed_df.printSchema())

    # get feature names dummy factors i.e for all "features" vector elements, since these columns names will not be stored in schema.
    #assembler_model = model.stages[-1]
    #ohe_feature_names = assembler_model.getFeatureNamesOut()
    #print(ohe_feature_names)

    # DUMP A SAMPLE SUMMARY on .1% data
    # ------------------------------------------------------------------------------------------------------------------
    summary_path = f"{out_path}/summary/{mode}"
    print("Saving summary of the data on: ", summary_path)
    summary = transformed_df.sample(withReplacement=False, fraction=0.001, seed=42).describe().coalesce(1)   # coalesce(1), generates a single summary.
    summary.write.mode("overwrite").option("header", "true").csv(summary_path)

    # OUTPUT DATA IN DIFFERENT SHARDS
    # ------------------------------------------------------------------------------------------------------------------
    # Different shards can be utilized by distributed training instead of saving a single file.
    # evenly randomize rows across num_shards partitions, then write
    print("Writing data on:", f"{out_path}/{mode}/")
    transformed_df.repartition(num_shards) \
                  .write.mode("overwrite") \
                  .parquet(f"{out_path}/{mode}/")
    print("Writing data Done on:", f"{out_path}/{mode}/")
    
    # STORE FEATURE NAMES METADATA  
    # ------------------------------------------------------------------------------------------------------------------
    if mode == "train":
        feature_metadata = {
            'target': target,
            'numerical_features': num_features,
            #'ohe_features': ohe_feature_names,
            'categorical_features': cat_features,
        }
    
        output_path = os.path.join(local_out_dir, "feature_metadata.json")
        with open(output_path, "w") as f:
            json.dump(feature_metadata, f, indent=2)

    print(f"Feature metadata written to {output_path}")
    
    print("Stored Data")
    spark.stop()