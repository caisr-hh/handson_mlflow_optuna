from models.sample_model import MlpModel
from configs.pipeline_config import DataConfig, ModelConfig

from data.data import EpochMetrics, TestMetrics, construct_data
import torch
import optuna




class Pipeline():

    def __init__(self):

        dataconfig = DataConfig()
        self.study = None
        self.data = construct_data(dataconfig)


    def run(self):

        ############################################################################################################
        #We pick a median pruner looking at the rolling median reported value and after the number of startup trials
        # and epochs, will start pruning

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        self.study =optuna.create_study(study_name="pruning study",direction="minimize", storage="sqlite:///optuna.db", load_if_exists=True, pruner = pruner)
        self.study.optimize(func = self.run_instance,n_trials = 30)
        ###########################################################################################################




    def run_instance(self, trial: optuna.Trial):
        #The objective function in this context.

        #######################################
        #Make suggestions from the active trial.
        config = ModelConfig(
            n_width = trial.suggest_int("n_width", low = 4, high =32),
            n_depth = trial.suggest_int("n_depth", low=0, high=3)
        )
        #######################################

        #Initiate model based on the suggested configuration
        self.model = MlpModel(config)

        #Send trial to training for pruning
        self.train(trial)

        # Run evaluation pipeline and get metrics
        metrics = self.evaluate()

        ######################################
        #Return the metric we are minimizing
        return metrics.test_loss
        ######################################



    def train(self, trial: optuna.Trial) -> MlpModel:
        # Basic train for x epochs, no traditional early stopping...

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.model.config.learning_rate
        )
        loss_function = torch.nn.BCELoss()

        self.model.train()

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

            ####################################################################
            #Report the training metrics, which competes against other trials at a given epoch.
            trial.report(metrics.epoch_loss, epoch)

            #If it falls behind, we stop.
            if trial.should_prune()==True:
                print(f"Trial {trial.number} is pruned at epoch {epoch}")
                break
            ####################################################################

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


        return metrics





pipeline = Pipeline()
pipeline.run()