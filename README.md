
# NYC Taxi Prediction At Scale
Data Exploration, Processing and ML Training at Scale using AWS Glue, Athena, and SageMaker for NYC Yellow Taxi Trip Records

This Project covers
- AWS Glue ETL Job for Schema unification of multiple parquet files for 5 years data i.e. 100-150M rows
- Athena for EDA 
- SageMaker PySparkProcessor for Data Processing 
- Sagemaker XGBoost Distributed Training and Evaluation

## Steps in Details:
- Manually Downloaded the dataset and uploaded to S3 bucket. 
- ETL Job & Pyspark script to unify the schema:
  - Craeted a Glue Crawler to detect the schema and create a table in the Glue Data Catalog. 
    - (Point to note: Parquet files have different schema for different file i.e. different datatypes for the same column. This will raise error when analysing with Athena. We can use pandas to load parquet files, correct the schema, and save them back to S3. But I will rather choose AW Glue Job.)
  - Created a Glue ETL Job to make schema consistent across all files i.e. same name for columns, datatypes across all files
  - Again crawl the cleaned data to create a new table in Glue Data Catalog so we can query it with Athena or Spark 
- EDA
  - Analyse data using Athena by querying the Glue Data Catalog table
- Data Processing using spark cluster 
  - A PySpark processor to transfom and preprocess the data 
  - The pyspark script reads data, cleans, transforms, summarizes and writes it back to S3
- Training and Evaluation
  - SageMaker XGBoost Distributed Training using the processed data
  - Evaluate the model performance

