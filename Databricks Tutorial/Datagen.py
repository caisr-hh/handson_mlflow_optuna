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

from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
pipeline_config = load_pipeline_config()

import data.data as data
data.generate_data_db(pipeline_config.data)