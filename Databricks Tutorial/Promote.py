# Databricks notebook source
# DBTITLE 1,Autoreload
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# DBTITLE 1,Cell 3
import numpy as np
from data.data import get_data_db
from misc.util import load_pipeline_config
from pipelines import Pipeline_Promote
config = load_pipeline_config()
pipeline = Pipeline_Promote(config)
result = pipeline.run()

dbutils.jobs.taskValues.set("result", result)