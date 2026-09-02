from dataclasses import asdict

import mlflow
import mlflow.pytorch
import torch
import yaml

from configs.pipeline_config import LoggerConfig, RunInfo
from misc.constants import REGISTERED_MODEL_NAME 
from misc.exceptions import HaltTraining
from misc.util import load_optuna_config
from data.data import EpochMetrics, TestMetrics, construct_data
import optuna
from torch.nn import Module
import matplotlib.pyplot as plt
from mlflow import MlflowClient
from optuna.integration import  get_current_trial

class Logger:
    def __init__(self, logger_config: LoggerConfig | None = None):
        self.config = logger_config
        self.parent = None
        self.runinfo = None


    def update_config(self, logger_config: LoggerConfig):
        self.config = logger_config

    def update_runinfo(self, runinfo : RunInfo):
        self.runinfo = RunInfo

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        # What to do with the results for each training epoch?
        pass

    def log_test(self, metrics: TestMetrics):
        # What to do with the test results?
        pass

    def log_study(self, study: optuna.study.Study):
        # What to do when training is finished
        pass

    def log_figure(self, fig):
        # Do we want to show or save the figure?
        pass

    def log_model(self, model: Module):
        # When the model is trained, what do we do with it?
        pass

    def log_interruption(self, context: str):
        # If the model training is interrupted, what do we do?
        pass
    def log_message(self, message : str, verobosity = 1):
        pass


class PipelineLogger(Logger):
    """
    This class propagates calls to all loggers in self.loggers.

    """

    def __init__(self,logger_config: LoggerConfig, loggers: dict[str, Logger] = {}):

        self.config = logger_config
        self.loggers = loggers
        if self.config is not None:
            for key in self.loggers.keys():
                self.loggers[key].parent = self
                self.loggers[key].update_config(logger_config)

    def set_logger(self, key: str, logger: Logger):
        self.loggers[key] = logger

    def reset_logger(self):
        self.loggers = {}


    def update_runinfo(self, runinfo : RunInfo):
        self.runinfo = RunInfo
        for key in self.loggers.keys():

            self.loggers[key].update_runinfo(runinfo)

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        for key in self.loggers.keys():

            self.loggers[key].log_epoch(metrics, epoch)

    def log_test(self, metrics: TestMetrics):
        for key in self.loggers.keys():

            self.loggers[key].log_test(metrics)

    def log_study(self, study: optuna.study.Study):
        for key in self.loggers.keys():

            self.loggers[key].log_study(study)

    def log_figure(self, fig):
        for key in self.loggers.keys():

            self.loggers[key].log_figure(fig)

    def log_model(self, model: Module):

        for key in self.loggers.keys():

            self.loggers[key].log_model(model)

    def log_interruption(self, context: str):
        # Collect and reraise exceptions, allowing all loggers to run their interruption handling.
        exceptions = []
        for key in self.loggers.keys():
            try:
                self.loggers[key].log_interruption(context)
            except Exception as error:
                exceptions.append(error)
        for exception in exceptions:
            raise exception

    def log_message(self, message : str, verobosity = 1):
        if verobosity >= self.config.verbosity:
            for key in self.loggers.keys():

                self.loggers[key].log_message(message, verobosity)


class LocalLogger(Logger):
    """
    The default logger.
    Prints epoch loss, test results..
    """

    def __init__(self, logger_config: LoggerConfig | None = None,show_figure: bool = True):
        self.parent = None
        self.config = logger_config
        self.show_figure = show_figure

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        print(
            f"Epoch {epoch}: Loss = {metrics.epoch_loss}, Accuracy = {metrics.epoch_accuracy},"
        )

    def log_test(self, metrics: TestMetrics):
        print(
            f"Final test loss = {metrics.test_loss}, Accuracy = {metrics.test_accuracy},"
        )

    def log_figure(self, fig):
        if self.show_figure:
            plt.show()
        
    def log_message(self, message : str, verobosity = 1):
        if verobosity >= self.config.verbosity:
            print(message)

class OptunaLogger(Logger):
    """
    The optuna logger will handle the pruning by monitoring the loss at each epoch and raising exceptions.
    """

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        # terminate the loop by first raising a generic HaltTraining interruption wit the pruning context.
        # It should then allow the other loggers to exit gracefully before reraising the interruption with the optuna specific error.

        trial = get_current_trial()
        if trial:
            # Report the loss to let the pruner decide if it is time to prune.
            trial.report(metrics.epoch_loss, epoch)

            # Should be prune?
            if trial.should_prune():
                # terminate the loop by first raising a generic HaltTraining interruption wit the "pruned" context.
                # It should then allow the other loggers to exit gracefully before reraising the interruption with the optuna specific
                raise HaltTraining(context="pruned")

    def log_interruption(self, context: str):
        # Raise the standard optuna.TrialPruned() error:
        if context == "pruned":
            raise optuna.TrialPruned()

    def log_model(self, model: Module):
        # Set the trial user attribute "mlflow_run_id" to trial, and add the model config to "config"
        runinfo = self.runinfo
        trial = runinfo.trial
        if trial:
            trial.set_user_attr("mlflow_run_id", runinfo.run_id)
            trial.set_user_attr("config", model.config.dict())


class MLFlowLogger(Logger):


    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        # convert the metrics into a dictionary using asdict() and log at step = epoch:
        mlflow.log_metrics(metrics=asdict(metrics), step=epoch)

    def log_test(self, metrics: TestMetrics):
        # Log test loss and accuracy:
        mlflow.log_metrics(asdict(metrics))

    def log_figure(self, fig):
        # The figure will reside in the root artifact storage for this run as boundary.png.
        mlflow.log_figure(fig, "boundary.png")

    def log_interruption(self, context: str):
        if context == "pruned":
            # Set the "status" tag to "pruned";
            mlflow.set_tag("status", "pruned")

    def log_model(self, model: Module):
        # Log the config dictionary:
        mlflow.log_params(model.config.dict())
        model_string = yaml.dump(model.config.model_dump())
        mlflow.log_text(model_string, artifact_file="configs/ModelConfig.yaml")
        # For grouping purposes we set the tag "status" as "complete":
        mlflow.set_tag("status", "complete")


class FinalLogger(MLFlowLogger):
    """
    This logger only needs a different log_model method to its parent class,
    turning the model into a scripted model with less source code and dependencies to handle.

    It then registers a tag in mlflow that identifies that this is the optimal

    This way we avoid bloating the registry, and we can register our first model.
    In this example we will promote it to the registry directly.
    """

    def log_model(self, model: Module):
        model_name = self.config.root + "_model"
        # Log the parameters as usual
        mlflow.log_params(model.config.dict())
        model_string = yaml.dump(model.config.model_dump())
        mlflow.log_text(model_string, artifact_file="configs/ModelConfig.yaml")



        # provide an input example to infer signature.
        input_example = data.get_table_db(model.config).test_loader.dataset[0:10][0].numpy()
        # Register our model. Use registered_model_name=REGISTERED_MODEL_NAME constant for the model registration.
        info = mlflow.pytorch.log_model(model, model_name,input_example)
        #Set a the "status" tag to "optimal", identifying this as the optimization winner
        client = MlflowClient()
        client.set_registered_model_alias(name=model_name, alias="contender", version=info.version)
        

