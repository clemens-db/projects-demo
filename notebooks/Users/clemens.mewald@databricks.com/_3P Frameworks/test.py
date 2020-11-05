# Databricks notebook source
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.spark
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK
from mlflow.models.signature import infer_signature


with mlflow.start_run(nested=True):
  #mlflow.sklearn.autolog()
  mlflow.spark.autolog()

  df = spark.table("clemens.flightdelays_augmented")
  pdf = df.toPandas()
  pdf = pdf[:1000]

  features_pd = pd.get_dummies(pdf[["Year", "Month", "DayofMonth", "DayOfWeek", "CRSDepTime", "CRSArrTime", "origin_prcp", "dest_prcp", "UniqueCarrier", "CRSElapsedTime", "Origin", "Dest", "Distance", "ArrDelay"]])

  X = features_pd.drop(['ArrDelay'], axis=1)
  y = features_pd['ArrDelay']
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

  rfr = RandomForestRegressor(max_depth=2, n_estimators=5)
  rfr.fit(X_train, y_train)

  y_pred = rfr.predict(X_test)
  rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
  mlflow.log_metric("rmse", rmse)

  mlflow.log_param("bootstrap", "True")
  mlflow.log_param("criterion", "mse")
  mlflow.log_param("max_features", "auto")
  mlflow.log_param("max_leaf_nodes", "None")
  mlflow.log_param("max_samples", "None")
  mlflow.log_param("min_impurity_decrease", "0.0")
  mlflow.log_param("min_impurity_split", "None")
  mlflow.log_param("min_samples_leaf", "1")
  mlflow.log_param("min_samples_split", "2")
  mlflow.log_param("min_weight_fraction_leaf", "0.0")
  mlflow.log_param("n_jobs", "None")
  mlflow.log_param("verbose", "0")
  mlflow.log_param("warm_start", "False")

  mlflow.set_tag("estimator_class", "sklearn.ensemble._forest.RandomForestRegressor")
  mlflow.set_tag("estimator_name", "RandomForestRegressor")
  mlflow.set_tag("sparkDatasourceInfo", "path=dbfs:/mnt/delta/flights/gold,version=4,format=delta")

  sig = infer_signature(X_train[:100], y_train[:100])
  mlflow.sklearn.log_model(rfr, 'model_signature', signature=sig, input_example=X_train.head(10), registered_model_name='2020-10-27_clemens_mewald@databricks_com_based on clemens_flightdelays_gold')
  
  import shap
  shap_values = shap.TreeExplainer(rfr).shap_values(X_train[:10])
  shap_plt = shap.summary_plot(shap_values, X_train[:10], plot_type="bar", show=False)
  
  import matplotlib.pyplot as plt
  plt.savefig('tree_explainer.png')
  
  mlflow.log_artifact('tree_explainer.png')


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from clemens.flightdelays_sample

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view tempFlightsWeatherA as select * from clemens.flightdelays_sample join clemens.airweather on
# MAGIC   clemens.flightdelays_sample.Origin = clemens.airweather.iata and
# MAGIC   clemens.flightdelays_sample.year = year(clemens.airweather.date) and
# MAGIC   clemens.flightdelays_sample.month = month(clemens.airweather.date) and
# MAGIC   clemens.flightdelays_sample.DayofMonth = day(clemens.airweather.date)

# COMMAND ----------

dff1 = spark.table('tempFlightsWeatherA')

dff1 = dff1.withColumnRenamed('prcp', 'origin_prcp')
dff1 = dff1.drop('date')
dff1 = dff1.drop('iata')

# COMMAND ----------

display(dff1)

# COMMAND ----------

dff1.createOrReplaceTempView("tempFlightsSample")

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace temporary view tempFlightsWeatherB as select * from tempFlightsSample join clemens.airweather on
# MAGIC   tempFlightsSample.Dest = clemens.airweather.iata and
# MAGIC   tempFlightsSample.year = year(clemens.airweather.date) and
# MAGIC   tempFlightsSample.month = month(clemens.airweather.date) and
# MAGIC   tempFlightsSample.DayofMonth = day(clemens.airweather.date)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tempFlightsWeatherB

# COMMAND ----------

dff2 = spark.table('tempFlightsWeatherB')

dff2 = dff2.withColumnRenamed('prcp', 'dest_prcp')
dff2 = dff2.drop('date')
dff2 = dff2.drop('iata')

# COMMAND ----------

display(dff2)

# COMMAND ----------

dff2.write.format("delta").save('/mnt/delta/clemens/airaugmented')
spark.sql("create table clemens.flightdelays_augmented using delta location '/mnt/delta/clemens/airaugmented'")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from clemens.flightdelays_augmented

# COMMAND ----------

