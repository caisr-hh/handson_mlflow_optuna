from dataclasses import asdict

import mlflow
import torch
import yaml

from configs.model_config import RunInfo
from demo.constants import REGISTERED_MODEL_NAME
from demo.exceptions import HaltTraining
from demo.data import EpochMetrics, TestMetrics, construct_data
import optuna
from torch.nn import Module
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, runinfo: RunInfo | None = None):

        self.runinfo = runinfo

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


class PipelineLogger(Logger):
    """
    This class propagates calls to all loggers in self.loggers.

    """

    def __init__(self, loggers: dict[str, Logger] = {}):

        self.loggers = loggers

    def set_logger(self, key: str, logger: Logger):
        self.loggers[key] = logger

    def reset_logger(self):
        self.loggers = {}

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


class LocalLogger(Logger):
    """
    The default logger.
    Prints epoch loss, test results and prints a figure if allowed (toggle off for batch runs).
    """

    def __init__(self, show_figure: bool = True):
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


class OptunaLogger(Logger):
    """
    The optuna logger will handle the pruning by monitoring the loss at each epoch and raising exceptions.
    """

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        # terminate the loop by first raising a generic HaltTraining interruption wit the pruning context.
        # It should then allow the other loggers to exit gracefully before reraising the interruption with the optuna specific error.

        trial = self.runinfo.trial
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
        if runinfo:
            trial.set_user_attr("mlflow_run_id", runinfo.run_id)
            trial.set_user_attr("config", model.config.dict())


class MLFlowLogger(Logger):

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        # convert the metrics into a dictionary using asdict() and log at step = epoch:
        metric_dict = asdict(metrics)
        # mlflow...
        pass

    def log_test(self, metrics: TestMetrics):
        # Log test loss and accuracy:
        metric_dict = asdict(metrics)
        # mlflow...
        pass

    def log_figure(self, fig):
        # Log as boundary.png.
        # mlflow...
        pass

    def log_interruption(self, context: str):
        if context == "pruned":
            # TODO: Set the "status" tag to "pruned";

            # mlflow...
            pass

    def log_model(self, model: Module):
        # TODO: Log the config dictionary as parameters:
        dict = model.config.dict()
        # mlflow...
        # TODO: Log the config as "configs/ModelConfig.yaml"
        model_string = yaml.dump(model.config.model_dump())
        # mlflow..
        # TODO: For grouping purposes we set the tag "status" as "complete":
        # mlflow...


class FinalLogger(MLFlowLogger):

    def log_model(self, model: Module):
        pass
