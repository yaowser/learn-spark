// Databricks notebook source
// MAGIC %md 
// MAGIC # Apache Spark on Databricks for Data Scientists
// MAGIC 
// MAGIC ** Welcome to Databricks! **
// MAGIC 
// MAGIC This notebook is intended to be the second step in your process to learn more about how to best use Apache Spark and Databricks together. We'll be walking through the core concepts, the fundamental abstractions, and some machine learning tools from the perspective of a data scientist! As a data scientist you're likely querying data from a variety of sources, joining together various tables and leveraging that information to build out statistical models. Databricks and Apache Spark make this process simple by allowing you to connect to a variety of sources and work with the data in one unified environment, then deploy your work to production quickly. Additionally because Databricks and Apache Spark support multiple languages, it's straightforward to do a lot of the analysis using the languages you know and the tools that you're likely already familiar with.
// MAGIC 
// MAGIC First, it's worth defining Databricks. Databricks is a managed platform for running Apache Spark - that means that you do not have to learn complex cluster management concepts nor perform tedious maintenance tasks to take advantage of Apache Spark. Databricks also provides a host of features to help its users be more productive with Spark. It's a point and click platform for those that prefer a user interface like data scientists or data analysts. This UI is accompanied by a sophisticated API for those that want to automate jobs and aspects of their data workloads. To meet the needs of enterprises, Databricks also includes features such as role-based access control and other intelligent optimizations that not only improve usability for users but also reduce costs and complexity for administrators.
// MAGIC 
// MAGIC ** The Gentle Introduction Series **
// MAGIC 
// MAGIC This notebook is a part of a series of notebooks aimed to get you up to speed with the basics of Spark quickly. This notebook is best suited for those that have very little or no experience with Spark. The series also serves as a strong review for those that have some experience with Apache Spark but aren't as familiar with some of the more sophisticated tools like UDF creation and machine learning pipelines. The other notebooks in this series are:
// MAGIC 
// MAGIC - [A Gentle Introduction to Apache Spark on Databricks](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html)
// MAGIC - [Apache Spark on Databricks for Data Scientists](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055194/484361/latest.html)
// MAGIC - [Apache Spark on Databricks for Data Engineers](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055109/484361/latest.html)
// MAGIC 
// MAGIC ## Tutorial Overview
// MAGIC 
// MAGIC This tutorial centers around a core idea that we hope to explore:
// MAGIC 
// MAGIC **The number of farmer's markets in a given zip code can be predicted from the income and taxes paid in a given area.**
// MAGIC 
// MAGIC It seems plausible that areas with higher income have more farmers markets simply because there is more of a market for those goods. Of course there are many potential holes in this idea, but that's part of the desire to test it :). This tutorial will explore our process for discovering whether or not we can accurately predict the number of farmer's markets! *It is worth mentioning that this notebook is not intended to be academically rigorous, but simply a good example of the work a data scientist might be performing using Apache Spark and Databricks.*
// MAGIC 
// MAGIC ## The Data
// MAGIC 
// MAGIC ![img](http://training.databricks.com/databricks_guide/USDA_logo.png)
// MAGIC 
// MAGIC The first of the two datasets that we will be working with is the **Farmers Markets Directory and Geographic Data**. This dataset contains information on the longitude and latitude, state, address, name, and zip code of farmers markets in the United States. The raw data is published by the Department of Agriculture. The version on the data that is found in Databricks (and is used in this tutorial) was updated by the Department of Agriculture on Dec 01, 2015.
// MAGIC 
// MAGIC ![img](http://training.databricks.com/databricks_guide/irs-logo.jpg)
// MAGIC 
// MAGIC The second dataset we will be working with is the **SOI Tax Stats - Individual Income Tax Statistics - ZIP Code Data (SOI)**. This study provides detailed tabulations of individual income tax return data at the state and ZIP code level and is provided by the IRS. This repository only has a sample of the data: 2013 and includes "AGI". The ZIP Code data shows selected income and tax items classified by State, ZIP Code, and size of adjusted gross income. Data is based on individual income tax returns filed with the IRS and is available for Tax Years 1998, 2001, 2004 through 2013. The data includes items, such as:
// MAGIC 
// MAGIC - Number of returns, which approximates the number of households
// MAGIC - Number of personal exemptions, which approximates the population
// MAGIC - Adjusted gross income
// MAGIC - Wages and salaries
// MAGIC - Dividends before exclusion
// MAGIC - Interest received
// MAGIC 
// MAGIC You can learn more about the two datasets on data.gov:
// MAGIC 
// MAGIC - [Farmer's Market Data](http://catalog.data.gov/dataset/farmers-markets-geographic-data/resource/cca1cc8a-9670-4a27-a8c7-0c0180459bef)
// MAGIC - [Zip Code Data](http://catalog.data.gov/dataset/zip-code-data)
// MAGIC 
// MAGIC ### Getting the Data
// MAGIC 
// MAGIC As a data scientist, your data is likely going to be living in a place like S3 or Redshift. Apache Spark provides simple and easy connectors to these data sources and Databricks provides simple demonstrations of how to use them. Just search in the Databricks guide (use the `?` at the top left) to see if your data source is available. For the purposes of this tutorial, our files are already available on S3 via `dbfs` or the Databricks file system. [While you're free to upload the csvs made available on data.gov as a table](https://docs.databricks.com/user-guide/tables.html#creating-tables) you can also (more easily) access this data via the `/databricks-datasets` directory which is a repository of public, Databricks-hosted datasets that is available on all Databricks accounts.

// COMMAND ----------



// COMMAND ----------

// MAGIC %md First things first! We've got to read in our data. This data is located in csv files so we'll use the [spark-csv](https://github.com/databricks/spark-csv) package to do this. In Databricks, it's as simple as specifying the format of the csv file and loading it in as a DataFrame. In Apache Spark 2.0, you do not need to use the `spark-csv` package and can just read the data in directly.
// MAGIC 
// MAGIC As a data scientist, you've likely come across DataFrames either in R, or python and pandas. Apache Spark DataFrames do not stray too much from this abstraction except that they are distributed across a cluster of machines instead of existing on one machine (as is typically the case with R or pandas). If you're not quite familiar with this, it might be worth reading through [a Gentle Introduction to Apache Spark on Databricks](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html).

// COMMAND ----------

val taxes2013 = sqlContext
  .read.format("com.databricks.spark.csv")
  .option("header", "true")
  .load("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")

// COMMAND ----------

// // in Apache Spark 2.0
// val taxes2013 = spark.read
//   .option("header", "true")
//   .csv("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")

// COMMAND ----------

val markets = sqlContext
  .read.format("com.databricks.spark.csv")
  .option("header", "true")
  .load("dbfs:/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/market_data.csv")

// COMMAND ----------

// // in Apache Spark 2.0
// val markets = spark.read
//   .option("header", "true")
//   .csv("dbfs:/databricks-datasets/data.gov/farmers_markets_geographic_data/data-001/market_data.csv")

// COMMAND ----------

// MAGIC %md Now that we've loaded in the data - let's go ahead and register the DataFrames as Spark SQL tables. 
// MAGIC 
// MAGIC While this might seem unnecessary, what it's going to allow you as a data scientist to do is to leverage your SQL skills immediately to manipulate the data. Some people prefer working directly with DataFrames while others prefer working in Spark SQL directly. Whatever the case, they take advantage of the same tungsten optimizations under the hood.

// COMMAND ----------

taxes2013.registerTempTable("taxes2013")
markets.registerTempTable("markets")

// COMMAND ----------

// // for spark 2.X
// taxes2013.createOrReplaceTempView("taxes2013")
// markets.createOrReplaceTempView("markets")

// COMMAND ----------

// MAGIC %md You can see that we are using the `registerTempTable`/`createOrReplaceTempView` method call to create this table. The lifetime of this temporary table is tied to the Spark/Spark SQL Context that was used to create this DataFrame. This means when you shutdown the SQLContext that is associated with a cluster (like when you shutdown the cluster) then the temporary table will disappear as well. In Databricks, Spark and Spark SQL Contexts are associated 1 to 1 with clusters.
// MAGIC 
// MAGIC ## Running SQL Commands
// MAGIC 
// MAGIC As we progress through the notebook, you'll notice that all SQL cells are prefaced with `%sql`. This tells the Databricks environment that you'd like to execute an SQL command. You can do the same with python and R as you will see in other tutorials and parts of the documentation.
// MAGIC 
// MAGIC In order to list the tables, we can show them very easily by simply executing `show tables`. You'll see that this also provides information about their lifetime (and whether or not they are temporary or not).

// COMMAND ----------

// MAGIC %sql show tables

// COMMAND ----------

// MAGIC %md Now that we've loaded in the data, let's take a quick look at it. The `display` command makes it easy to quickly display a subset of the table that we have. This operates directly on our `DataFrame`.

// COMMAND ----------

display(taxes2013)

// COMMAND ----------

// MAGIC %md A roughly equivalent operation would be to do a `SELECT *` on our recently created temp table above just like above. You'll see that this automatically gets limited to the first 1000 rows in order to avoid overflowing the browser.

// COMMAND ----------

// MAGIC %sql SELECT * FROM taxes2013

// COMMAND ----------

// MAGIC %md We can see that we've got a variety of columns that you might want to look into further however for the purpose of this analysis I'm only going to look at a very small subset. I'm also going to perform two small manipulations to this data:
// MAGIC 
// MAGIC 1. I'm going to do some simple type conversions and rename the columns to something a bit more semantic so that it's easier to talk about them going forward. 
// MAGIC 2. I'm also going to shorten each zip code to be four digits instead of 5. This will make it so that we look a bit more at the general location around a zip code as opposed to a very specific one. This is an imprecise overall process, but for the purpose of this example works just fine.

// COMMAND ----------

// MAGIC %sql 
// MAGIC DROP TABLE IF EXISTS cleaned_taxes;
// MAGIC 
// MAGIC CREATE TABLE cleaned_taxes AS
// MAGIC SELECT state, int(zipcode / 10) as zipcode, 
// MAGIC   int(mars1) as single_returns, 
// MAGIC   int(mars2) as joint_returns, 
// MAGIC   int(numdep) as numdep, 
// MAGIC   double(A02650) as total_income_amount,
// MAGIC   double(A00300) as taxable_interest_amount,
// MAGIC   double(a01000) as net_capital_gains,
// MAGIC   double(a00900) as biz_net_income
// MAGIC FROM taxes2013

// COMMAND ----------

// MAGIC %md See how easy it is to create a derivative table to work with? Now that the data is cleaned up. I can start exploring the data. To do that I'm going to leverage the Databricks environment to create some nice plots. 
// MAGIC 
// MAGIC First I'm going to explore the average total income per zip code per state. We can see here that on the whole there isn't anything drastic. New Jersey and California have higher average incomes per zip code.

// COMMAND ----------

// MAGIC %sql select * FROM cleaned_taxes

// COMMAND ----------

// MAGIC %md Another way that we could perform that same operation is through the DataFrame API as is shown below. This shows the true power of Apache Spark - a lot of the concepts and operations that you know readily transfer from other languages/DataFrame abstractions like pandas and R.

// COMMAND ----------

// commented out so that we don't have duplicate plots!

// val cleanedTaxes = sqlContext.table("cleaned_taxes")
// display(cleanedTaxes.groupBy("state").avg("total_income_amount"))

// COMMAND ----------

// MAGIC %md Next let's explore some specifics of this particular dataset. First we'll do a simple describe, just as we might do in R or pandas.

// COMMAND ----------

display(cleanedTaxes.describe())

// COMMAND ----------

// MAGIC %md  Let's look at the set of zip codes with the lowest total capital gains and plot the results. You can see that we're able to use simple expressive SQL to achieve these results in a very straightforward manner as well as some familiar DataFrame manipulations available in R and Python. 

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT zipcode, SUM(net_capital_gains) AS cap_gains
// MAGIC FROM cleaned_taxes 
// MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
// MAGIC GROUP BY zipcode
// MAGIC ORDER BY cap_gains ASC
// MAGIC LIMIT 10

// COMMAND ----------

// MAGIC %md There we are, we've performed some basic analysis of our data and taken advantage of the simple plotting made available in Databricks. However I'm still a bit intrigued, we've taken a look at capital gains however one of our other columns refers to business net income. Let's look at a combination of capital gains and business net income to see what we find. It's worth stressing again how simple it is to iteratively build up these queries with Spark SQL as well - it's just so simple!
// MAGIC 
// MAGIC In the below query, I've built this `combo` metric that represents the total capital gains and business net income by zip code. This is weighted very strongly by capital gains as we can see in the plot.

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT zipcode, 
// MAGIC   SUM(biz_net_income) as business_net_income, 
// MAGIC   SUM(net_capital_gains) as capital_gains, 
// MAGIC   SUM(net_capital_gains) + SUM(biz_net_income) as capital_and_business_income
// MAGIC FROM cleaned_taxes 
// MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
// MAGIC GROUP BY zipcode
// MAGIC ORDER BY capital_and_business_income DESC
// MAGIC LIMIT 50

// COMMAND ----------

// MAGIC %md While these plots have been quick to produce within Databricks - let's take a moment to review Apache Spark's execution model. Run the above code and click on the `view` button. That will bring up the DAG or the directed acyclic graph of the Apache Spark tasks and stages that need to be performed in order to create this graph. Remember that Apache Spark only executes transformations when we perform actions. We introduced this concept in [the gentle introduction to Apache Spark and Databricks notebook](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html) and it's worth reviewing if this does not seem familiar.
// MAGIC 
// MAGIC We can also get a peak at what *will* happen when we use the `EXPLAIN` keyword in SQL.

// COMMAND ----------

// MAGIC %sql
// MAGIC EXPLAIN 
// MAGIC   SELECT zipcode, 
// MAGIC     SUM(biz_net_income) as net_income, 
// MAGIC     SUM(net_capital_gains) as cap_gains, 
// MAGIC     SUM(net_capital_gains) + SUM(biz_net_income) as combo
// MAGIC   FROM cleaned_taxes 
// MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
// MAGIC   GROUP BY zipcode
// MAGIC   ORDER BY combo desc
// MAGIC   limit 50

// COMMAND ----------

// equivalent to the above
sqlContext.sql("""
  SELECT zipcode, 
    SUM(biz_net_income) as net_income, 
    SUM(net_capital_gains) as cap_gains, 
    SUM(net_capital_gains) + SUM(biz_net_income) as combo
  FROM cleaned_taxes 
  WHERE NOT (zipcode = 0000 OR zipcode = 9999)
  GROUP BY zipcode
  ORDER BY combo desc
  limit 50""").explain

// COMMAND ----------

// MAGIC %md We can see above that we first fetch the data from `dbfs:/user/hive/warehouse/cleaned_taxes` which is where the data is stored when we registered it as a temporary table. After that a variety of filters and aggregates are performed. These are highly optimized to be performed efficiently and reduce the data that has to be sent around the cluster.
// MAGIC 
// MAGIC However we can do even better. One thing that is great about Apache Spark is that out of the box it can store and access tables in memory. All that we need to do is to `cache` the data to do so. We can either do this directly in SQL (at which point the cache will be done *eagerly* or right away), or we can do it through the `sqlContext` with the `cacheTable` method which will be performed lazily.

// COMMAND ----------

sqlContext.cacheTable("cleaned_taxes")

// COMMAND ----------

// MAGIC %sql CACHE TABLE cleaned_taxes

// COMMAND ----------

// MAGIC %md Now that we've cached it, let's go ahead and run the exact same query again. You'll notice that it takes just a fraction of the time because the data is stored in memory.

// COMMAND ----------

// MAGIC %sql
// MAGIC SELECT zipcode, 
// MAGIC   SUM(biz_net_income) as net_income, 
// MAGIC   SUM(net_capital_gains) as cap_gains, 
// MAGIC   SUM(net_capital_gains) + SUM(biz_net_income) as combo
// MAGIC FROM cleaned_taxes 
// MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
// MAGIC GROUP BY zipcode
// MAGIC ORDER BY combo desc
// MAGIC limit 50

// COMMAND ----------

// MAGIC %md This is revealed when we look at the `EXPLAIN` plan as well - you'll see that instead of going down to the source data it performs an `InMemoryColumnarTableScan`. This means that it has all of the information that it needs in memory so Apache Spark can avoid doing a lot of work. While the speed up is rather small with this dataset, the benefits that we can realize with much larger data are definitely significant!

// COMMAND ----------

// MAGIC %sql
// MAGIC EXPLAIN 
// MAGIC   SELECT zipcode, 
// MAGIC     SUM(biz_net_income) as net_income, 
// MAGIC     SUM(net_capital_gains) as cap_gains, 
// MAGIC     SUM(net_capital_gains) + SUM(biz_net_income) as combo
// MAGIC   FROM cleaned_taxes 
// MAGIC   WHERE NOT (zipcode = 0000 OR zipcode = 9999)
// MAGIC   GROUP BY zipcode
// MAGIC   ORDER BY combo desc
// MAGIC   limit 50

// COMMAND ----------

// MAGIC %md Now that we've spent some time exploring the IRS data - let's take a moment to look at the Farmer's Market Data. We'll start off with a total summation of farmer's markets per state. You'll notice that I'm not using SQL at this time. While we can certainly query the SQL table, it's worth showing that all the functions available in SQL are also available directly on a DataFrame.

// COMMAND ----------

display(markets.groupBy("State").count())

// COMMAND ----------

// MAGIC %md 
// MAGIC While these datasets probably warrant a lot more exploration, let's go ahead and prep the data for use in Apache Spark MLLib. Apache Spark MLLib has some specific requirements about how inputs are structured. Firstly, input data has to be numeric unless you're performing a transformation inside of a data pipeline. What this means for you as a user is that Apache Spark won't automatically convert string to categories for instance, instead the output will be a `Double` type. Let's go ahead and prepare our data so that it meets those requirements as well as joining together our input data with the target variable - the number of farmer's markets in a given zipcode.

// COMMAND ----------

val cleanedTaxes = sqlContext.sql("SELECT * FROM cleaned_taxes")

val summedTaxes = cleanedTaxes
  .groupBy("zipcode")
  .sum() // because of AGI, where groups income groups are broken out 

val cleanedMarkets = markets
  .selectExpr("*", "int(zip / 10) as zipcode")
  .groupBy("zipcode")
  .count()
  .selectExpr("double(count) as count", "zipcode as zip")
// selectExpr is short for Select Expression - equivalent to what we
// might be doing in SQL SELECT expression

val joined = cleanedMarkets
  .join(summedTaxes, cleanedMarkets("zip") === summedTaxes("zipcode"), "outer")

// COMMAND ----------

// MAGIC %md Now that we've joined our tax data to our output variable, we're going to have to do a final bit of cleanup before we can input this data into Spark MLLib. For example, when we go to display our joined data, we're going to have null values. 

// COMMAND ----------

display(joined)

// COMMAND ----------

// MAGIC %md Currently Apache Spark MLLib doesn't allow us to enter in null values (nor would it make sense to leave them out). Therefore we're going to replace them with 0's. Luckily, DataFrames make it easy to work with null data under the `.na` prefix as you'll see below. [You can see all of the null functions in the API documentation.](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrameNaFunctions) These should be very familiar and similar to what you might find in pandas or in R DataFrames.

// COMMAND ----------

val prepped = joined.na.fill(0)
display(prepped)

// COMMAND ----------

// MAGIC %md Now that all of our data is prepped. We're going to have to put all of it into one column of a vector type for Spark MLLib. This makes it easy to embed a prediction right in a DataFrame and also makes it very clear as to what is getting passed into the model and what isn't without having to convert it to a numpy array or specify an R formula. This also makes it easy to incrementally add new features, simply by adding to the vector. In the below case rather than specifically adding them in, I'm going to create an exclusionary group and just remove what is NOT a feature.

// COMMAND ----------

val nonFeatureCols = Array("zip", "zipcode", "count")
val featureCols = prepped.columns.diff(nonFeatureCols)

// COMMAND ----------

// MAGIC %md Now I'm going to use the `VectorAssembler` in Apache Spark to Assemble all of these columns into one single vector. To do this I'll have to set the input columns and output column. Then I'll use that assembler to transform the prepped data to my final dataset.

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

val finalPrep = assembler.transform(prepped)

// COMMAND ----------

// MAGIC %md Now let's take a look at our feature columns all graphed out with one another.

// COMMAND ----------

display(finalPrep.drop("zip").drop("zipcode").drop("features"))

// COMMAND ----------

// MAGIC %md Now in order to follow best practices, I'm going to perform a random split of 70-30 on the dataset for training and testing purposes. This can be used to create a validation set as well however this tutorial will omit doing so. It's worth noting that MLLib also supports performing hyperparameter tuning with cross validation and pipelines. All this can be found in [the Databrick's Guide](https://docs.databricks.com).

// COMMAND ----------

val Array(training, test) = finalPrep.randomSplit(Array(0.7, 0.3))

// Going to cache the data to make sure things stay snappy!
training.cache()
test.cache()

println(training.count())
println(test.count())

// COMMAND ----------

// MAGIC %md 
// MAGIC # Apache Spark MLLib
// MAGIC 
// MAGIC Now we're going to get into the core of Apache Spark MLLib. At a high level, we're going to create an instance of a `regressor` or `classifier`, that in turn will then be trained and return a `Model` type. Whenever you access Spark MLLib you should be sure to import/train on the name of the algorithm you want as opposed to the `Model` type. For example:
// MAGIC 
// MAGIC You should import:
// MAGIC 
// MAGIC `org.apache.spark.ml.regression.LinearRegression`
// MAGIC 
// MAGIC as opposed to:
// MAGIC 
// MAGIC `org.apache.spark.ml.regression.LinearRegressionModel`
// MAGIC 
// MAGIC In the below example, we're going to use linear regression.
// MAGIC 
// MAGIC The linear regression that is available in Spark MLLib supports an elastic net parameter allowing you to set a threshold of how much you would like to mix l1 and l2 regularization, for [more information on Elastic net regularization see Wikipedia](https://en.wikipedia.org/wiki/Elastic_net_regularization).
// MAGIC 
// MAGIC As we saw above, we had to perform some preparation of the data before inputting it into the model. We've got to do the same with the model itself. We'll set our hyper parameters, print them out and then finally we can train it! The `explainParams` is a great way to ensure that you're taking advantage of all the different hyperparameters that you have available.

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

val lrModel = new LinearRegression()
  .setLabelCol("count")
  .setFeaturesCol("features")
  .setElasticNetParam(0.5)

println("Printing out the model Parameters:")
println("-"*20)
println(lrModel.explainParams)
println("-"*20)

// COMMAND ----------

// MAGIC %md Now finally we can go about fitting our model! You'll see that we're going to do this in a series of steps. First we'll fit it, then we'll use it to make predictions via the `transform` method. This is the same way you would make predictions with your model in the future however in this case we're using it to evaluate how our model is doing. We'll be using regression metrics to get some idea of how our model is performing, we'll then print out those values to be able to evaluate how it performs.

// COMMAND ----------

import org.apache.spark.mllib.evaluation.RegressionMetrics
val lrFitted = lrModel.fit(training)

// COMMAND ----------

// MAGIC %md Now you'll see that since we're working with exact numbers (you can't have 1/2 a farmer's market for example), I'm going to check equality by first rounding the value to the nearest digital value.

// COMMAND ----------

val holdout = lrFitted
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "count", 
    """CASE double(round(prediction)) = count 
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout)

// COMMAND ----------

// MAGIC %md Now let's see what proportion was exactly correct.

// COMMAND ----------

display(holdout.selectExpr("sum(equal)/sum(1)"))

// COMMAND ----------

// MAGIC %md Let's also calculate some regression metrics.

// COMMAND ----------

// have to do a type conversion for RegressionMetrics
val rm = new RegressionMetrics(
  holdout.select("prediction", "count").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm.meanSquaredError)
println("MAE: " + rm.meanAbsoluteError)
println("RMSE Squared: " + rm.rootMeanSquaredError)
println("R Squared: " + rm.r2)
println("Explained Variance: " + rm.explainedVariance + "\n")

// COMMAND ----------

// MAGIC %md I found these results to be sub-optimal, so let's try exploring another way to train the model. Rather than training on a single model with hard-coded parameters, let's train using a [pipeline](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.Pipeline). 
// MAGIC 
// MAGIC A pipeline is going to give us some nice benefits in that it will allow us to use a couple of transformations we need in order to transform our raw data into the prepared data for the model but also it provides a simple, straightforward way to try out a lot of different combinations of parameters. This is a process called [hyperparameter tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization) or grid search. To review, grid search is where you set up the exact parameters that you would like to test and MLLib will automatically create all the necessary combinations of these to test.
// MAGIC 
// MAGIC For example, below we'll set `numTrees` to 20 and 60 and `maxDepth` to 5 and 10. The parameter grid builder will automatically construct all the combinations of these two variable (along with the other ones that we might specify too). Additionally we're also going to use [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) to tune our hyperparameters, this will allow us to attempt to try to control [overfitting](https://en.wikipedia.org/wiki/Overfitting) of our model.
// MAGIC 
// MAGIC Lastly we'll need to set up a [Regression Evaluator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.evaluation.RegressionEvaluator) that will evaluate the models that we choose based on some metric (the default is RMSE). The key take away is that the pipeline will automatically optimize for our given metric choice by exploring the parameter grid that we set up rather than us having to do it manually like we would have had to do above.
// MAGIC 
// MAGIC Now we can go about training our random forest! 
// MAGIC 
// MAGIC *note: this might take a little while because of the number of combinations that we're trying and limitations in workers available.*

// COMMAND ----------

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.{Pipeline, PipelineStage}

val rfModel = new RandomForestRegressor()
  .setLabelCol("count")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
  .addGrid(rfModel.maxDepth, Array(5, 10))
  .addGrid(rfModel.numTrees, Array(20, 60))
  .build()
// Note, that this parameter grid will take a long time
// to run in the community edition due to limited number
// of workers available! Be patient for it to run!
// If you want it to run faster, remove some of
// the above parameters and it'll speed right up!

val steps:Array[PipelineStage] = Array(rfModel)

val pipeline = new Pipeline().setStages(steps)

val cv = new CrossValidator() // you can feel free to change the number of folds used in cross validation as well
  .setEstimator(pipeline) // the estimator can also just be an individual model rather than a pipeline
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(new RegressionEvaluator().setLabelCol("count"))

val pipelineFitted = cv.fit(training)

// COMMAND ----------

// MAGIC %md Now we've trained our model! Let's take a look at which version performed best!

// COMMAND ----------

println("The Best Parameters:\n--------------------")
println(pipelineFitted.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))
pipelineFitted
  .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
  .stages(0)
  .extractParamMap

// COMMAND ----------

// MAGIC %md Now let's take a look at our holdout set results.

// COMMAND ----------

val holdout2 = pipelineFitted.bestModel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction)) as prediction", 
    "count", 
    """CASE double(round(prediction)) = count 
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout2)

// COMMAND ----------

// MAGIC %md As well as our regression metrics on the test set.

// COMMAND ----------

// have to do a type conversion for RegressionMetrics
val rm2 = new RegressionMetrics(
  holdout2.select("prediction", "count").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm2.meanSquaredError)
println("MAE: " + rm2.meanAbsoluteError)
println("RMSE Squared: " + rm2.rootMeanSquaredError)
println("R Squared: " + rm2.r2)
println("Explained Variance: " + rm2.explainedVariance + "\n")

// COMMAND ----------

// MAGIC %md Finally we'll see an improvement in our "exactly right" proportion as well!

// COMMAND ----------

display(holdout2.selectExpr("sum(equal)/sum(1)"))

// COMMAND ----------

// MAGIC %md 
// MAGIC # Conclusion
// MAGIC 
// MAGIC We can see from the above that we identified a fairly significant link by leveraging the pipeline, a more sophisticated model, and better hyperparameter tuning. However these results are still a bit disappointing. With that being said, we're working with very few features and we've likely made some assumptions that just aren't quite valid (like the zip code shortening). Also just because a rich zip code exists doesn't mean that the farmer's market would be held in that zip code too. In fact we might want to start looking at neighboring zip codes or doing some sort of distance measure to predict whether or not there exists a farmer's market in a certain mile radius from a wealthy zip code.
// MAGIC 
// MAGIC With that being said, we've got a lot of other potential features and plenty of other parameters to tune on our random forest so play around with the above pipeline and see if you can improve it further!
// MAGIC 
// MAGIC Doing so is outside of the context of this notebook! Now go out there and start making predictions!
