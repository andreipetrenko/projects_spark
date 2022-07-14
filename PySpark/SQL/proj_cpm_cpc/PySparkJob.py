import io
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max, min, datediff
from pyspark.sql import functions as func


def process(spark, input_file, target_path):
    # прочитаем переданный файл и сформируем DataFrame
    df = spark.read.parquet(input_file)

    # изменим структуру DataFrame
    dff = df.groupBy('ad_id', 'target_audience_count', 'has_video', 'ad_cost', 'ad_cost_type').agg(
        min(col('date')).alias('minDate'), max(col('date')).alias('maxDate'),
        func.count(func.when(col('event') == 'view', True)).alias('view'),
        func.count(func.when(col('event') == 'click', True)).alias('click'))
    dff = dff.withColumn('is_cpm', (col('ad_cost_type') == 'CPM').cast('integer'))
    dff = dff.withColumn('is_cpc', (col('ad_cost_type') == 'CPC').cast('integer'))
    dff = dff.withColumn('day_count', datediff(col('maxDate'), col('minDate')))
    dff = dff.withColumn('CTR', ((col('click') / col('view')) * 100).cast('double'))
    dff = dff.drop('date', 'time', 'platform', 'event', 'compaign_union_id', 'client_union_id', 'ad_cost_type',
                   'minDate', 'maxDate', 'view', 'click')
    dff = dff.dropDuplicates()

    # разбиваем DataFrame на train и test
    dff_train, dff_test = dff.randomSplit([0.75, 0.25])

    #сохраняем результат
    dff_train.coalesce(1).write.parquet(target_path+'/train')
    dff_test.coalesce(1).write.parquet(target_path+'/test')


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
