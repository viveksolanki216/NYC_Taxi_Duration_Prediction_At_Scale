#-------------------------
# 0. Objective
#-------------------------
# Parquet files have different dtypes for the same columns, that needs to be corrected and kept same for al files

import sys
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql.types import *
from pyspark.sql.functions import col, lit 
from awsglue.utils import getResolvedOptions
import boto3

#-----------------------------
# 1. Parse Command Line Args
#-----------------------------
args = getResolvedOptions(sys.argv, ['input_path', 'output_path'])
input_path = args['input_path']
output_path = args['output_path']

#-------------------------------
# 2. Set up Spark/Glue Context
#-------------------------------
sc = SparkContext()
glueContext = GlueContext(sc) 
spark = glueContext.spark_session 

#-------------------------------
# 3. Define Unified Schema
#-------------------------------
unified_schema = StructType([
    StructField("vendorid", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("ratecodeid", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("pulocationid", IntegerType(), True),
    StructField("dolocationid", IntegerType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("congestion_surcharge", DoubleType(), True),
    StructField("airport_fee", DoubleType(), True),
    StructField("cbd_congestion_fee", DoubleType(), True) # total 20 fields
])

#-------------------------------
# 4. List all parquet files inside the s3_uri_path
#-------------------------------
s3 = boto3.client("s3")
bucket, key = input_path.replace("s3://", "").split("/",1)
#print(bucket,key)
response = s3.list_objects_v2(Bucket=bucket, Prefix=key)
#print(response)
files = [f"s3://{bucket}/{obj['Key']}" for obj in response['Contents'] if obj['Key'].endswith('.parquet')]

#-------------------------------
# 5. Manually cast each column of each parquet file
#-------------------------------
# Need to manually cast each column for each parquet
dfs = []
for path in files:
    df = spark.read.parquet(path)
    df = df.toDF(*[c.lower() for c in df.columns])

    for field in unified_schema.fields:
        if field.name in df.columns:
            df = df.withColumn(field.name, df[field.name].cast(field.dataType))
        else:
            df = df.withColumn(field.name, lit(None).cast(field.dataType)) # If missing column
            
    df = df.select([f.name for f in unified_schema.fields])
    dfs.append(df)
    
# Merge/Concate the list of dfs to one single dataframe
# Make sure all dataframes are of same columns
df_harmonized = dfs[0]
for df in dfs[1:]:
    df_harmonized = df_harmonized.unionByName(df)

#-------------------------------
# 6. To partition the data daily
#-------------------------------
from pyspark.sql.functions import to_date
df_harmonized = df_harmonized.withColumn("tpep_pickup_date", to_date("tpep_pickup_datetime"))

#-------------------------------
# 7. Write harmonized data
#-------------------------------
# this will partition the data by each data and keep data in separeate date folders that will speed up the time-based queries.
df_harmonized.write.mode("overwrite") \
    .partitionBy("tpep_pickup_date") \
    .parquet(output_path)

print(f"Harmonization complete. Output written to: {output_path}")





























