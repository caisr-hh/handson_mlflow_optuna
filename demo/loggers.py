from dataclasses import asdict

import mlflow
import torch

from configs.model_config import RunInfo
from demo.constants import REGISTERED_MODEL_NAME
from demo.exceptions import HaltTraining
from demo.sample_model import ModelConfig
from demo.data import EpochMetrics, TestMetrics, construct_data
from pydantic import BaseModel
import optuna
from torch.nn import Module
from demo.sample_model import MlpModel
import matplotlib.pyplot as plt



class Logger:
    def __init__(self, runinfo : RunInfo | None=None):

        self.runinfo=runinfo



    def log_epoch(self, metrics: EpochMetrics,epoch: int):
        #What to do with the results for each training epoch?
        pass

    def log_test(self, metrics: TestMetrics):
        #What to do with the test results?
        pass

    def log_study(self, study: optuna.study.Study):
        #What to do when training is finished
        pass

    def log_figure(self, fig):
        #Do we want to show or save the figure?
        pass

    def log_model(self, model: Module):
        #When the model is trained, what do we do with it?
        pass

    def log_interruption(self, context: str):
        #If the model training is interrupted, what do we do?
        pass


class PipelineLogger(Logger):
    """
    This class propagates calls to all loggers in self.loggers.

    """

    def __init__(self, loggers: dict[str, Logger] = {}):

        self.loggers = loggers

    def set_logger(self,key : str, logger : Logger):
        self.loggers[key]=logger
    def reset_logger(self):
        self.loggers={}

    def log_epoch(self, metrics: EpochMetrics,epoch: int):
        for key in self.loggers.keys():

            self.loggers[key].log_epoch(metrics,epoch)

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
        #Collect and reraise exceptions, allowing all loggers to run their interruption handling.
        exceptions=[]
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
    Prints epoch loss, test results and saves a figure.
    """
    def __init__(self, show_figure: bool =True):
        self.show_figure=show_figure
    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        print(
            f"Epoch {epoch}: Loss = {metrics.epoch_loss}, Accuracy = {metrics.epoch_accuracy},"
        )

    def log_test(self, metrics: TestMetrics):
        print(f"Final test loss = {metrics.test_loss}, Accuracy = {metrics.test_accuracy},")


    def log_figure(self, fig):
        if self.show_figure:
            plt.show()




class OptunaLogger(Logger):


    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        trial=self.runinfo.trial
        if trial:
            trial.report(metrics.epoch_loss,epoch)

            if trial.should_prune():
                raise HaltTraining(context="pruned")

    def log_interruption(self, context: str):

        if context=="pruned":
            raise optuna.TrialPruned()

    def log_model(self,model:Module):
        runinfo=self.runinfo
        trial = runinfo.trial
        if runinfo:
            trial.set_user_attr("mlflow_run_id", runinfo.run_id)
            trial.set_user_attr("config", model.config.dict())


class MLFlowLogger(Logger):

    def log_epoch(self, metrics: EpochMetrics, epoch: int):
        mlflow.log_metrics(metrics=asdict(metrics), step=epoch)


    def log_test(self, metrics: TestMetrics):
        mlflow.log_metrics(asdict(metrics))


    def log_figure(self, fig):
        mlflow.log_figure(fig, "boundary.png")


    def log_interruption(self, context: str):
        if context=="pruned":
            mlflow.set_tag("status","pruned")
    def log_model(self, model: Module):

        mlflow.set_tag("status", "complete")

class FinalLogger(MLFlowLogger):

    def log_model(self, model: Module):
        mlflow.set_tag("status", "optimal")
        script_model = torch.jit.script(model)
        data_example=construct_data(model.config).test_loader.dataset[0:10][0].numpy()
        mlflow.pytorch.log_model(
            pytorch_model=script_model,
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=data_example,
        )

