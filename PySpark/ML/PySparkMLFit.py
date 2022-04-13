import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

# Используйте как путь куда сохранить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, train_data, test_data):
    train = spark.read.parquet(train_data, header=True, inferSchema=True)
    test = spark.read.parquet(test_data, header=True, inferSchema=True)

    features = VectorAssembler(
        inputCols=['target_audience_count', 'has_video', 'is_cpm', 'is_cpc', 'ad_cost', 'day_count'],
        outputCol='features')

    dtr = DecisionTreeRegressor(labelCol='ctr', featuresCol='features')
    rfr = RandomForestRegressor(labelCol='ctr', featuresCol='features')
    gbtr = GBTRegressor(labelCol='ctr', featuresCol='features')

    pipelineDtr = Pipeline(stages=[features, dtr])
    pipelineRfr = Pipeline(stages=[features, rfr])
    pipelineGbtr = Pipeline(stages=[features, gbtr])

    paramGridDtr = ParamGridBuilder() \
        .addGrid(dtr.maxDepth, [2, 3, 4, 5, 6, 9]) \
        .build()
    paramGridRfr = ParamGridBuilder() \
        .addGrid(rfr.maxDepth, [2, 3, 4, 5, 6, 9]) \
        .addGrid(rfr.numTrees, [3, 6, 9, 12, 15, 18, 21]) \
        .build()
    paramGridGbtr = ParamGridBuilder() \
        .addGrid(rfr.maxDepth, [2, 3, 4, 5, 6, 9]) \
        .addGrid(rfr.numTrees, [3, 6, 9, 12, 15, 18, 21]) \
        .build()

    evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction", metricName="rmse")

    crossvalDtr = CrossValidator(estimator=pipelineDtr,
                                 estimatorParamMaps=paramGridDtr,
                                 evaluator=evaluator,
                                 numFolds=2)
    crossvalRfr = CrossValidator(estimator=pipelineRfr,
                                 estimatorParamMaps=paramGridRfr,
                                 evaluator=evaluator,
                                 numFolds=2)
    crossvalGbtr = CrossValidator(estimator=pipelineGbtr,
                                 estimatorParamMaps=paramGridGbtr,
                                 evaluator=evaluator,
                                 numFolds=2)

    modelDtr = crossvalDtr.fit(train)
    modelRfr = crossvalRfr.fit(train)
    modelGbtr = crossvalGbtr.fit(train)

    predictionsDrt = modelDtr.transform(test)
    predictionsRfr = modelRfr.transform(test)
    predictionsGbtr = modelGbtr.transform(test)

    rmseDrt = evaluator.evaluate(predictionsDrt)
    rmseRfr = evaluator.evaluate(predictionsRfr)
    rmseGbtr = evaluator.evaluate(predictionsGbtr)

    print("RMSE DRT - " + str(rmseDrt))
    print("RMSE RFR - " + str(rmseRfr))
    print("RMSE GBTR - " + str(rmseGbtr))

    if rmseDrt < rmseRfr:
        if rmseDrt < rmseGbtr:
            rmse = rmseDrt
            modelDtr.bestModel.write().overwrite().save(MODEL_PATH)
        else:
            rmse = rmseGbtr
            modelGbtr.bestModel.write().overwrite().save(MODEL_PATH)
    else:
        if rmseRfr < rmseGbtr:
            rmse = rmseRfr
            modelRfr.bestModel.write().overwrite().save(MODEL_PATH)
        else:
            rmse = rmseGbtr
            modelGbtr.bestModel.write().overwrite().save(MODEL_PATH)

    print("MIN RMSE - "+str(rmse))


def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)

