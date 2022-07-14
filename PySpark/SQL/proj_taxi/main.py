import os
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, DoubleType

# поля справочника
dim_columns = ['id', 'name']

payment_rows = [
    (1, 'Credit card'),
    (2, 'Cash'),
    (3, 'No charge'),
    (4, 'Dispute'),
    (5, 'Unknown'),
    (6, 'Voided trip'),
]

trips_schema = StructType([
    StructField('vendor_id', StringType(), True),
    StructField('tpep_pickup_datetime', TimestampType(), True),
    StructField('tpep_dropoff_datetime', TimestampType(), True),
    StructField('passenger_count', IntegerType(), True),
    StructField('trip_distance', DoubleType(), True),
    StructField('ratecode_id', IntegerType(), True),
    StructField('store_and_fwd_flag', StringType(), True),
    StructField('pulocation_id', IntegerType(), True),
    StructField('dolocation_id', IntegerType(), True),
    StructField('payment_type', IntegerType(), True),
    StructField('fare_amount', DoubleType(), True),
    StructField('extra', DoubleType(), True),
    StructField('mta_tax', DoubleType(), True),
    StructField('tip_amount', DoubleType(), True),
    StructField('tolls_amount', DoubleType(), True),
    StructField('improvement_surcharge', DoubleType(), True),
    StructField('total_amount', DoubleType(), True),
    StructField('congestion_surcharge', DoubleType()),
])


def create_dict(spark: SparkSession, header: list[str], data: list) -> DataFrame:
    """создание словаря"""
    df = spark.createDataFrame(data=data, schema=header)
    return df


def create_datamark(spark: SparkSession) -> DataFrame:
    # data_path = "https://storage.yandexcloud.net/s3petrenko/2020/yellow_tripdata_2020-04.csv"
    data_path = os.path.join(Path(__name__).parent, './data', '*.csv')

    trip_fact = spark.read \
        .option("header", "true") \
        .schema(trips_schema) \
        .csv(data_path)

    datamart = trip_fact \
        .where(trip_fact['vendor_id'].isNotNull()) \
        .filter(f.to_date(trip_fact['tpep_pickup_datetime']).between('2020-04-01', '2020-04-30')) \
        .groupBy(trip_fact['payment_type'],
                 f.to_date(trip_fact['tpep_pickup_datetime']).alias('dt')
                 ) \
        .agg(f.avg(trip_fact['total_amount']).alias('avg_trip_cost'), \
             f.avg(trip_fact['trip_distance']).alias("avg_trip_dist")) \
        .select(f.col('payment_type'),
                f.col('dt'),
                f.col('avg_trip_cost'),
                f.col('avg_trip_dist')) \
        .orderBy(f.col('dt').desc(), f.col('payment_type'))

    return datamart


def main(spark: SparkSession):
    payment_dim = create_dict(spark, dim_columns, payment_rows)

    payment_dim.show()

    datamart = create_datamark(spark).cache()
    # datamart.show(truncate=False, n=100)

    joined_datamart = datamart \
        .join(other=payment_dim, on=payment_dim['id'] == f.col('payment_type'), how='inner') \
        .select(payment_dim['name'].alias('payment_name').alias('Payment type'),
                f.col('dt').alias('Date'),
                f.round(f.col('avg_trip_cost'), 2).alias('Average trip cost'),
                f.round(f.col('avg_trip_dist'), 2).alias('Avg trip km cost'),
                )

    #joined_datamart.show(truncate=False, n=100)
    joined_datamart.write.mode('overwrite').csv('output')


if __name__ == '__main__':
    main(SparkSession
         .builder
         .appName('Spark job for taxis data')
         .getOrCreate())

