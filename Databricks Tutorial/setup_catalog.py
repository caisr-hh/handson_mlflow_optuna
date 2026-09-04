# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC **Run the SQL cell below to create the catalogues and schemas we will use in this example:**

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE CATALOG ecml;
# MAGIC CREATE SCHEMA ecml.models;
# MAGIC CREATE SCHEMA ecml.data;