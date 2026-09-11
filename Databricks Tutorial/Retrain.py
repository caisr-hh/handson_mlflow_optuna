# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "-r '/Workspace/Shared/ECML WIP/Databricks Tutorial/requirements.txt'",
# ]
# ///
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import logging
logging.getLogger('mlflow.tracking.context.registry').setLevel(logging.ERROR)

from pipelines import Pipeline_Retrain


from configs.pipeline_config import PipelineConfig
import mlflow

config = PipelineConfig.from_file("configs/pipeline_config.yaml")

pipeline = Pipeline_Retrain(config)
pipeline.run()

