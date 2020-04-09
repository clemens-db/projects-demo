# Databricks notebook source
import mlflow

with mlflow.start_run():
  mlflow.log_metric('mname', 3.0)

# COMMAND ----------

# test 2