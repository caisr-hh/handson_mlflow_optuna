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

import logging
logging.getLogger('mlflow.tracking.context.registry').setLevel(logging.ERROR)

from pipelines import Pipeline_HPO
from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
import mlflow


config = load_pipeline_config()




pipeline = Pipeline_HPO(config)
winner = pipeline.run()



# COMMAND ----------

dbutils.widgets.text("result", "default")

status = dbutils.widgets.get("result")
print("Hellow world:")

# COMMAND ----------

from configs.pipeline_config import RunInfo

info = RunInfo()
info.trial

# COMMAND ----------

from mlflow.optuna.storage import MlflowStorage
import optuna
import mlflow
from pipelines import Pipeline_HPO
from misc.util import load_pipeline_config

config = load_pipeline_config()
storage = MlflowStorage(experiment_id = mlflow.get_experiment_by_name(config.logger.experiment_hpo).experiment_id)

study = optuna.study.load_study(study_name=config.logger.experiment_hpo,storage=storage)
