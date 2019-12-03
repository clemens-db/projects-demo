# Databricks notebook source
import mlflow

with mlflow.start_run():
  mlflow.log_metric('mname', 2.0)

# COMMAND ----------

