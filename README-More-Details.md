
# NYC Taxi Prediction At Scale
Data Exploration, Processing and ML Training at Scale using AWS Glue, Athena, and SageMaker for NYC Yellow Taxi Trip Records


## 0. Download the data. 
- Manually download the dataset from the [NYC gov link](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Or write a script as "Download_Data.py", but it's not working.

Data will be in different parquet files at month level.

### Parquet files
- columnar storage format, i.e. you can read only the columns you need.
- Binary format (Non-readable)
- efficient, fast, takes less memory for large datasets.

## 3. Glue Job: for ETL (Extract, Transform, Load) and Schema Unification
#### AWS Glue Job 
- Built for scalable ETL (Extract, Transform, Load) over large datasets stored in S3. 
  - Extract: Read data from S3 (or other sources like RDS, Redshift, etc.)
  - Transform: Clean, flatten, join, filter the data.
  - Load: Write the transformed data back to S3 (or other destinations i.e. RDS, Redshift, etc.)
- Glue uses Apache Spark under the hood, which is distributed and much faster for cleaning data across many files.
- Use Pyspark to write a script that reads the parquet files, cleans the data (manually cast each column to a unified schema), and writes it back to S3.
- Serverless, can scale automatically, no need to manage infrastructure.

#### Other Options to do above:
- PySparkProcessor in SageMaker: 
  - It can run PySpark scripts on distrbuted data and spark cluster.
  - It abstracts Spark cluster management and runs distributed spark jobs.

- EMR (Elastic MapReduce):
  - AWS managed Hadoop/Spark cluster.
  - More control over Spark cluster, but requires more setup and management.
  - Good for very large datasets or complex Spark jobs.
 
| Feature               | AWS Glue Job             | SageMaker PySparkProcessor        | EMR with Spark                          |
| --------------------- | ------------------------ | --------------------------------- | --------------------------------------- |
| Spark Engine          | ✅ Yes                    | ✅ Yes                             | ✅ Yes                                   |
| Cluster Management    | Serverless (AWS-managed) | Ephemeral SageMaker-managed Spark | You manage (manual/automated)           |
| Best For              | ETL and data prep        | Feature engineering for ML        | Large, complex data pipelines           |
| Integration           | Data Catalog, S3         | SageMaker Pipeline, Feature Store | HDFS, S3, Hive Metastore                |
| Startup Time          | Medium                   | Medium                            | Slow (if persistent), fast (if running) |
| Autoscaling           | ✅                        | ❌ (fixed instance count)          | ✅ (with config)                         |
| Custom Configurations | Limited                  | Limited                           | Fully configurable                      |
| Cost                  | Pay per job              | Pay per job                       | Pay per hour (Spot OK)                  |
| Control & Flexibility | Low                      | Medium                            | High                                    |

## 4.  EDA
When data is large and couldn't fit in memory i.e. pandas dataframe, we can use:
 - Glue Studio Notebooks with QuickSight for EDA
 - AWS Athena to query the data in S3 using SQL.
 - Sagemaker Studio notebook with PySparkMagic (Need to connect to Glue)
   -Studio noteboos need to connect to a spark cluster.
### **AWS Athena**: 
Serverless query service to analyze data in S3 using SQL.
How Athena Works with S3: Athena lets you run SQL queries directly on S3 files in formats like: CSV, JSON, Parquet (columnar + compressed, best performance),  ORC, Avro, Gzipped data
It’s completely serverless, so You don’t provision infrastructure You pay per query, per amount of data scanned

### Glue Crawler:
- A Glue Crawler is a tool that automatically scans your raw data in S3 (or other sources), detects the schema, and creates a table in the Glue Data Catalog. 
- Scans multiple files and creates a single table if they have the same schema.
- It doesn't create the table in a traditional database, but the metadata/schema is stored in the Glue Data Catalog, which Athena can query. Like a pointer to the s3 data.
- Might need to update schema, as Glue Crawler might not detect all columns correctly, especially if the data is in multiple files.

How to Use Athena on S3 Files (Steps)
- Create a Glue Crawler (Once):
  - Go to AWS Glue > Crawlers, enter the required details.
  - There will be a Glue Service Role created automatically, which has permissions to read from S3 and write to the Glue Data Catalog. Add following permissions to the role:
        - AWSGlueConsoleFullAccess
        - CloudWatchLogsFullAccess
  - It will create a single table for all the parquet files in the specified S3 bucket.
- Query via Athena 
  - Go to Athena > Query Editor, and: 
- Without Glue (Advanced Option), You can create the table manually in data catalog using SQL. But using a Glue Crawler is easier and avoids schema mistakes.
    
## 5. Data Processing: PySpark Processor
Pandas is memory-bound and runs on a single machine, while PySpark can handle large datasets, scales horizontally, and 
runs distributed computations across multiple instances. So we will use PySparkProcessor that brings its own container
that has pyspark and other dependencies pre-installed, abstracts the Spark cluster management, and runs distributed 
spark jobs.

#### PySparkProcessor is a SageMaker SDK class that lets you:
- Run PySpark (Python + Spark) scripts.
- Leverage distributed processing over multiple instances.
- Handle large datasets (like multiple large Parquet/CSV files in S3).
- Use Spark’s power for transformations, joins, filtering, aggregations, etc.

#### When to Use PySparkProcessor
- Dataset doesn’t fit in memory (Pandas fails).
- You need to process or transform data in parallel.
- You’re working with millions of rows / large Parquet files.
- You want the benefits of Apache Spark without managing Spark clusters.

#### Why not use ScriptProcessor or SKLearnProcessor?

## 6. Model Training using Sagemaker distributed training



