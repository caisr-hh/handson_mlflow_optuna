# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "mlflow>=3.1",
# ]
# ///
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
pipeline_config = load_pipeline_config()



# COMMAND ----------

from pipelines import Pipeline_Evaluator

pipeline_evaluator = Pipeline_Evaluator(pipeline_config)
result = pipeline_evaluator.run()

dbutils.jobs.taskValues.set("result", result)

# COMMAND ----------

result