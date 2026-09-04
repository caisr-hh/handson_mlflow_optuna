# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "-r '/Workspace/Users/mikael.andersson@hh.se/ECML WIP/Tutorial steps 1/requirements.txt'",
# ]
# ///
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import mlflow

mymodel = mlflow.pyfunc.load_model("models:/ecml.models.ecml_model@contender")

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