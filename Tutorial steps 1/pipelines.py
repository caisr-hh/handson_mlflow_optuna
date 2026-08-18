from configs.pipeline_config import PipelineConfig
import torch
from models.sample_model import MlpModel
from data.data import EpochMetrics, TestMetrics, construct_data
from logs.logger import PipelineLogger, Logger
from util import load_pipeline_config



class TrainingPipeline():

    def __init__(self, config: PipelineConfig, loggers: dict[str, Logger]):
        self.config = config
        self.logger = PipelineLogger(loggers)

    def run(self):
        self.data = construct_data(self.config.data)
        self.model = MlpModel(self.config.model)
        self.train()
        self.evaluate()
        self.save()
        pass



    def evaluate(self):
        pass

    def save(self):
        pass


    def train(self) -> MlpModel:

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.model.config.learning_rate
        )
        loss_function = torch.nn.BCELoss()

        self.model.train()
        #try:
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
        #except HaltTraining as error:
        #    self.logger.log_interruption(error.context)

        return self.model

    def evaluate(self):
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

        return metrics