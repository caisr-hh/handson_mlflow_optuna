# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "-r '/Workspace/Shared/ECML WIP/Databricks Tutorial/requirements.txt'",
# ]
# ///
# DBTITLE 1,Autoreload
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Cell 3
import numpy as np
from data.data import get_data_db
from misc.util import load_pipeline_config
from pipelines import Pipeline_Deploy
config = load_pipeline_config()
pipeline = Pipeline_Deploy(config)
result = pipeline.run()

dbutils.jobs.taskValues.set("result", result)