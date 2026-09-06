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

# MAGIC %md
# MAGIC # Hyperparameter optimization
# MAGIC _See pipelines.py for definitions of pipeline components, configs/PipelineConfig(.py/.yaml) for configurations and logs.logger.py for logging utilities._
# MAGIC
# MAGIC This part of the pipeline handles the optuna optimization. We focus on the parameters depth vs width. With our task, we expect it to gravitate towards exploring regions with a high amount of layers and a big layer width to solve the problem optimally.
# MAGIC
# MAGIC We use the mlflow storage backend, meaning that the study stores it's information as mlflow runs which can be inspected in the UI directly. The study is tied to the experiment and is consequently removed when the experiment is deleted. For the purpose of this demo we recreate the study from scratch for each run of the HPO, keeping the runs themselves for inspection in mlflow. 
# MAGIC
# MAGIC We store the config of the optimal model in optuna, which can then be fetched when training future models.
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run hyperparameter optimization
import logging
logging.getLogger('mlflow.tracking.context.registry').setLevel(logging.ERROR)

from pipelines import Pipeline_HPO
from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
import mlflow


config = load_pipeline_config()




pipeline = Pipeline_HPO(config)
winner = pipeline.run()

