import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{Imputer, IndexToString, StringIndexer, StringIndexerModel, PCA, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object KickStarter {
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("KickStarter"))

    if(args.length < 3) {
      println("Usage:")
      println("spark-submit --class \"KickStarter\" `input dataset` `output folder` `metrics output folder`")
    }

    val spark = SparkSession
      .builder()
      .appName("KickStarter")
      .getOrCreate()

    val projects = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .csv(args(0))
      .toDF()

    // Drop columns not used for prediction
    val trimmedProjects = projects.drop(
      "main_category",
      "currency",
      "deadline",
      "goal",
      "launched",
      "pledged",
      "usd pledged",
      "usd_pledged_real"
    )

    // Filter out rows that have empty values
    val cleanProjects = trimmedProjects.filter(
      trimmedProjects("category").notEqual("") &&
      trimmedProjects("state").notEqual("") &&
      trimmedProjects("country").notEqual("") &&
      trimmedProjects("usd_goal_real").notEqual("") &&
      trimmedProjects("backers").notEqual("")
    )

    // Convert the datatype of columns from String to Double
    val convertedProjects = cleanProjects
      .withColumn("goal", cleanProjects("usd_goal_real").cast(DoubleType))
      .withColumn("numBackers", cleanProjects("backers").cast(DoubleType))
      .drop("usd_goal_real", "backers")

    // Separate the live projects from the finished ones
    val finishedProjects = convertedProjects.filter(convertedProjects("state").notEqual("live"))
    val liveProjects = convertedProjects.filter(convertedProjects("state").equalTo("live"))

    // Index Categorical Features
    val categoryIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .setHandleInvalid("keep")

    val countryIndexer = new StringIndexer()
      .setInputCol("country")
      .setOutputCol("countryIndex")
      .setHandleInvalid("keep")

    val stateIndexer = new StringIndexer()
      .setInputCol("state")
      .setOutputCol("stateIndex")
      .setHandleInvalid("keep")

    // Replace NaN values with Mean of that column
    val imputer = new Imputer()
      .setInputCols(Array("categoryIndex", "countryIndex", "goal", "numBackers"))
      .setOutputCols(Array("categoryIndexImputed", "countryIndexImputed", "goalImputed", "numBackersImputed"))

    // Merge all the feature columns to a single column
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("categoryIndexImputed", "countryIndexImputed", "goalImputed", "numBackersImputed"))
      .setOutputCol("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")

    val randomForestClassifier = new RandomForestClassifier()
      .setLabelCol("stateIndex")
      .setFeaturesCol("pcaFeatures")

    // Convert label indices to labels
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedState")
      .setLabels(stateIndexer.fit(finishedProjects).labels)

    // Specify pipeline with stages
    val pipeline = new Pipeline()
      .setStages(Array(
        categoryIndexer,
        countryIndexer,
        stateIndexer,
        imputer,
        vectorAssembler,
        pca,
        randomForestClassifier,
        labelConverter
      ))

    // Specify parameter grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.minInfoGain, Array(0, 0.1, 0.2))
      .addGrid(randomForestClassifier.numTrees, Array(10, 15))
      .addGrid(pca.k, Array(2, 3))
      .build()

    // Specify cross validator for parameter tuning
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(
        new MulticlassClassificationEvaluator()
          .setLabelCol("stateIndex")
      )
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    // Split the dataset into train-test
    val Array(training, test) = finishedProjects.randomSplit(Array(0.7, 0.3))

    // Fit the model to the training data
    val crossValidatorModel = crossValidator.fit(training)

    // Test the model on test data
    val classification = crossValidatorModel.transform(test)

    // Evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("stateIndex")
      .setPredictionCol("prediction")

    evaluator.setMetricName("accuracy")
    val accuracy = evaluator.evaluate(classification)

    evaluator.setMetricName("weightedPrecision")
    val weightedPrecision = evaluator.evaluate(classification)

    evaluator.setMetricName("weightedRecall")
    val weightedRecall = evaluator.evaluate(classification)

    evaluator.setMetricName("f1")
    val f1 = evaluator.evaluate(classification)

    val evaluationMetrics = sc.parallelize(List(("Accuracy", accuracy), ("Weighted Precision", weightedPrecision), ("Weighted Recall", weightedRecall), ("F1", f1)))
    evaluationMetrics.saveAsTextFile(args(2))

    // Predict the status of the live projects
    val predictions = crossValidatorModel.transform(liveProjects)

    val output = predictions.select(
      predictions("ID"),
      predictions("name"),
      predictions("category"),
      predictions("predictedState")
    ).collect().mkString("\n")

    sc.parallelize(List(s"Predictions\n" + output)).saveAsTextFile(args(1))
  }
}
