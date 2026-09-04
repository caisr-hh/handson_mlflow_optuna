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

from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
pipeline_config = load_pipeline_config()

from pipelines import Pipeline_Evaluator

pipeline_evaluator = Pipeline_Evaluator(pipeline_config)
result = pipeline_evaluator.run()


#Sets a task value for this notebook, used for flow control in databricks jobs!
dbutils.jobs.taskValues.set("result", result)