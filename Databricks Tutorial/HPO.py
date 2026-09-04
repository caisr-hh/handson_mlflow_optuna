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

import logging
logging.getLogger('mlflow.tracking.context.registry').setLevel(logging.ERROR)

from pipelines import Pipeline_HPO
from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
import mlflow


config = load_pipeline_config()




pipeline = Pipeline_HPO(config)
winner = pipeline.run()

