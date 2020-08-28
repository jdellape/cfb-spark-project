from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F  
from pyspark.sql import udf
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel  

if __name__ == "__main__":
	
	conf = SparkConf().setAppName("CFBExcitementClassifier").set("spark.hadoop.yarn.resourcemanager.address", "spark01:8032")
	sc = SparkContext(conf=conf)
	spark = SparkSession(sc)

	games_df = spark.read.csv("/user/jdellape/input/games.csv", header = True)

	bet_lines_df = spark.read.csv("/user/jdellape/input/betting_lines.csv", header = True)
	bet_lines_df = bet_lines_df[['id','lines']]

	#Join the dataframes together
	df = games_df.join(bet_lines_df, on=['id'], how='inner')

	df = df.select(col("away_conference"), col("away_team"), col("conference_game"), col("excitement_index"), col("home_conference"), col("home_team"), col("id"), col("lines").alias("bet_lines"))

	df = df.dropna()

	#Filter out where bet_lines is functionally empty, but not technically null
	df = df.filter("bet_lines != '[]'")

	#Make a sequence of splits to isolate just the spread number

	df = df.withColumn("new_col", df.bet_lines)

	df = df.withColumn("new_col", split(col("new_col"), "'spread'").getItem(1))
	df = df.withColumn("new_col", split(col("new_col"), ",").getItem(0))
	df = df.withColumn("new_col", split(col("new_col"), "'").getItem(1))

	df = df.withColumnRenamed("new_col", "spread")

	df.drop('bet_lines').show()

	df = df.filter("spread != 'None'")

	#Remove the bet_lines column from df
	df = df[['away_conference','away_team','conference_game','excitement_index','home_conference','home_team','id','spread']]

	df = df.select(col("away_team"), col("home_team"), concat(col("away_team"), lit(" | "), col("home_team")).alias("matchup"), col("id").alias("game_id"),
					col("excitement_index").cast(FloatType()).alias("excitement_index"), col("spread").cast(FloatType()).alias("spread"))


	def classify_excitement(x):
		if x < 5.0:
			return 0
		else:
			return 1

	udf_classify_excitement = F.udf(classify_excitement, IntegerType())
	df = df.withColumn("high_excitement", udf_classify_excitement("excitement_index"))


	#Load in plays data
	plays_df = spark.read.csv("/user/jdellape/input/plays.csv", header = True)

	#Define lists for pass types and rush types
	pass_types = ['Pass Reception', 'Pass Interception Return', 'Pass Incompletion', 'Sack', 'Passing Touchdown', 'Interception Return Touchdown']
	rush_types = ['Rush', 'Rushing Touchdown']

	#Define functions I am going to use

	def flag_pass_types(x):
		if x not in pass_types:
			return 0
		else:
			return 1

	def flag_rush_types(x):
		if x not in rush_types:
			return 0
		else:
			return 1

	udf_flag_pass_types = F.udf(flag_pass_types, IntegerType())
	udf_flag_rush_types = F.udf(flag_rush_types, IntegerType())

	plays_df = plays_df.withColumn("pass_play", udf_flag_pass_types("play_type"))
	plays_df = plays_df.withColumn("rush_play", udf_flag_rush_types("play_type"))

	plays_grouped = plays_df.groupBy("game_id").sum("pass_play","rush_play")

	plays_grouped = plays_grouped.withColumnRenamed("sum(pass_play)", "pass_plays")
	plays_grouped = plays_grouped.withColumnRenamed("sum(rush_play)", "rush_plays")

	plays_grouped = plays_grouped.withColumn('pass_rush_ratio', plays_grouped['pass_plays'] / plays_grouped['rush_plays'])

	#Join our play data into our main df

	df = df.join(plays_grouped, on=['game_id'], how='inner')

	df.show(10)



	#Prep data for modeling

	final_columns = ['pass_rush_ratio','spread', 'matchup', 'high_excitement']

	
	(training_data, test_data) = df[final_columns].randomSplit([0.7, 0.3], seed=11)

	#Encode team matchup string
	matchup_indexer = StringIndexer(inputCol='matchup', outputCol='indexedMatchup', handleInvalid='keep')
	matchup_encoder = OneHotEncoder(inputCol='indexedMatchup', outputCol='matchupVec')


	assembler = VectorAssembler(inputCols=['spread','pass_rush_ratio', 'matchupVec'],outputCol="features")
	 
	# Train a RandomForest model.

	rf = RandomForestClassifier(labelCol="high_excitement", featuresCol="features", numTrees=150)

	pipeline = Pipeline(stages=[matchup_indexer, matchup_encoder, assembler, rf])
 
	# Train model.  This also runs the indexers.
	model = pipeline.fit(training_data)

	va = model.stages[-2]
	classifier = model.stages[-1]

	print(list(zip(va.getInputCols(), classifier.featureImportances)))

	# Predictions
	predictions = model.transform(test_data)
	print("Printing schema of predictions: ")
	predictions.printSchema()
 
	# Show information regarding predictions
	predictions.select("prediction", "high_excitement", "features").show(25)	
 
	# Select (prediction, true label) and compute test error
	predictionsAndLabels = predictions.select(col("prediction"), col("high_excitement").cast(DoubleType()))
	predictionsAndLabels = predictionsAndLabels.rdd

	metrics = BinaryClassificationMetrics(predictionsAndLabels)

	# Area under precision-recall curve
	print("Area under PR = %s" % metrics.areaUnderPR)

	# Area under ROC curve
	print("Area under ROC = %s" % metrics.areaUnderROC)
