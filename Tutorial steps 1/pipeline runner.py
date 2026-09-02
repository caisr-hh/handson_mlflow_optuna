# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "mlflow>=3.0 ",
# ]
# ///
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,ell
import logging
logging.getLogger('mlflow.tracking.context.registry').setLevel(logging.ERROR)

from pipelines import HPO_Pipeline, Retrain_Pipeline
from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
import mlflow


config = load_pipeline_config()




pipeline = HPO_Pipeline(config)
winner = pipeline.run()
retrain = Retrain_Pipeline(winner)
retrain.run()



# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

with mlflow.start_run():

    mlflow.log_param("test", 1)

    mlflow.log_metric("accuracy", 0.95)

# COMMAND ----------

import mlflow

print(mlflow.__version__)

# COMMAND ----------

import tempfile