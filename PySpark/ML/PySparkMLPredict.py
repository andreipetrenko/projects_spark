import io
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Используйте как путь откуда загрузить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, input_file, output_file):
    test_data = spark.read.parquet(input_file, header=True, inferSchema=True)
    persModel = PipelineModel.load(MODEL_PATH)
    predictions = persModel.transform(test_data)
    predictions.select(['ad_id', 'prediction']).write.csv(output_file)


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    output_file = argv[1]
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
