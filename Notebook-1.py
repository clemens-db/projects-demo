# Databricks notebook source
import mlflow

with mlflow.start_run():
  mlflow.log_metric('mname', 1.0)

# COMMAND ----------

