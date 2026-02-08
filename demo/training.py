import optuna

from demo.plots import plot_res
import torch

from demo.sample_model import MlpModel, ModelConfig
from demo.data import EpochMetrics, TestMetrics, construct_data
from configs.model_config import RunInfo
from demo.exceptions import HaltTraining
from demo.loggers import PipelineLogger


class Pipeline:
    def __init__(self, config: ModelConfig, logger: PipelineLogger):
        super(Pipeline, self).__init__()
        self.model = None
        self.data = None
        self.logger = logger
        self.config = config
        self.runinfo = RunInfo()
        self.test_metrics = None

    def run(self):

        # Initialize model
        self.data = construct_data(self.config)
        self.model = MlpModel(self.config)

        # Train the model:
        self.training_loop().eval()

        # Get the test result:
        self.test_model()
        self.logger.log_test(self.test_metrics)

        # Plot decision boundary if desired:
        fig = plot_res(model=self.model, data=self.data)
        self.logger.log_figure(fig)

        # Log the model:
        self.logger.log_model(self.model)
        return self.model

    def training_loop(self) -> MlpModel:

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.model.config.learning_rate
        )
        loss_function = torch.nn.BCELoss()

        self.model.train()
        try:
            for epoch in range(self.model.config.epoch_max):

                loss_sum = 0
                n_correct = 0
                n_samples = 0
                for batch_input, batch_labels in self.data.training_loader:

                    optimizer.zero_grad()
                    prediction = self.model(batch_input)
                    loss = loss_function(prediction, batch_labels)
                    classifications = prediction > 0.5
                    n_correct += sum(classifications == batch_labels.bool()).item()
                    n_samples += len(classifications)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss.item()

                loss_sum = loss_sum / len(self.data.training_loader)
                accuracy = n_correct / n_samples
                metrics = EpochMetrics(epoch_loss=loss_sum, epoch_accuracy=accuracy)

                self.logger.log_epoch(
                    metrics, epoch
                )  # Calls all our included loggers for this epoch
        except HaltTraining as error:
            self.logger.log_interruption(error.context)

        return self.model

    def test_model(self):
        self.model.eval()
        loss_function = torch.nn.BCELoss()
        loss_sum = 0
        n_correct = 0
        n_samples = 0
        for batch_input, batch_labels in self.data.test_loader:
            prediction = self.model(batch_input)
            loss = loss_function(prediction, batch_labels)
            classifications = prediction > 0.5
            n_correct += sum(classifications == batch_labels.bool()).item()
            n_samples += len(classifications)

            loss_sum += loss.item()
        loss_sum = loss_sum / len(self.data.test_loader)
        accuracy = n_correct / n_samples
        metrics = TestMetrics(test_loss=loss_sum, test_accuracy=accuracy)
        self.test_metrics = metrics
        # Log test perf.

        return metrics
