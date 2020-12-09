# Databricks notebook source
print ('hello world')

# COMMAND ----------

import mlflow

with mlflow.start_run():
  mlflow.log_metric('auc', 1.0)

# COMMAND ----------

