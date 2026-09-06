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
# MAGIC # Evaluation
# MAGIC _See pipelines.py for definitions of pipeline components, configs/PipelineConfig(.py/.yaml) for configurations and logs.logger.py for logging utilities._
# MAGIC
# MAGIC As our first step in the small demo pipeline we will do some sanity checks that decides if:
# MAGIC
# MAGIC - The current champion model is within acceptable thresholds (compared to its original performance) without needing retraining (result = 0).
# MAGIC - The current champions perfomance is degraded on the current data, a sufficient test loss increase over the test loss logged during training. Try retraining based on the optimal study parameters (result = 1).
# MAGIC - No study matching the pipeline configs description exists yet, do initial HPO with optuna (result = 2).
# MAGIC
# MAGIC The result is set as a task variable to help direct the full databricks job accordingly. 
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Run evaluation component
from misc.util import load_pipeline_config

from configs.pipeline_config import PipelineConfig
pipeline_config = load_pipeline_config()

from pipelines import Pipeline_Evaluator

pipeline_evaluator = Pipeline_Evaluator(pipeline_config)
result = pipeline_evaluator.run()


#Sets a task value for this notebook, used for flow control in databricks jobs!
dbutils.jobs.taskValues.set("result", result)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC In Pipeline_Evaluator We use the tag "champion" to fetch the currently deployed model from mlflow:
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC `model = mlflow.pytorch.load_model(model_uri=f"models:/{self.config.logger.model_name_uc}@{"champion"}")`
# MAGIC
# MAGIC `client = MlflowClient()`
# MAGIC
# MAGIC `info = client.get_model_version_by_alias(self.config.logger.model_name_uc, "champion")`
# MAGIC
# MAGIC
# MAGIC  We inspect the run and fetch the registered models original training metrics:
# MAGIC
# MAGIC `run = client.get_run(info.run_id)`
# MAGIC
# MAGIC `old_metrics = run.data.metrics`
# MAGIC
# MAGIC Then we run the evaluation procedure shared between the pipeline components and compare it with the threshold defined in the pipeline config.``