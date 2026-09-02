
#Tutorial repo---------------------------------
from configs.pipeline_config import PipelineConfig, RunInfo
from models.sample_model import MlpModel
from data.data import EpochMetrics, TestMetrics, construct_data, get_data_db
from logs.logger import PipelineLogger, Logger
import logs.logger as loggers
from misc.util import load_pipeline_config
from misc.exceptions import HaltTraining
#----------------------------------------------

# MLFLOW--------------------------------------
import mlflow
from mlflow.optuna.storage import MlflowStorage
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException, RestException
#----------------------------------------------

#Optuna----------------------------------------
import optuna
#----------------------------------------------

#Torch-----------------------------------------
import torch
#----------------------------------------------

#Misc------------------------------------------
from pathlib import Path
from datetime import datetime
import tempfile
#----------------------------------------------



class Pipeline():

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(loggers = {}, logger_config = config) #Empty logger by default
        self.study = None
        self.data = construct_data(self.config.data)
    def run(self):

        local_loggers = {
            "local":loggers.LocalLogger(),
            }
        self.logger=PipelineLogger(loggers = local_loggers, logger_config = self.config.logger)
        with mlflow.start_run(nested = false) as run:
            runinfo = RunInfo(
                run_id = mlflow.active_run().info.run_id
            )
            self.logger.update_runinfo(runinfo)
            self.run_instance()
    

    
    def run_instance(self):
        

        try:
            self.model = MlpModel(self.config.model)
            self.train()
            metrics = self.evaluate()
            
            self.save(self.model,metrics)
            self.logger.log_test(metrics)

        except HaltTraining as error:
            print("Training halted...")
        return metrics



    def train(self) -> MlpModel:
        #Shared training process.

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
    


    def evaluate(self):
        #Shared evaluation process
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
    

    def save(self, model, metrics, samples = None):

        uid = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        model_name = f"MLP_{uid}"
        model_path = path.Path("checkpoints") / model_name
        
        model_path.makedir(parents = true, exist_ok = True)
        
        pass
  

class Pipeline_HPO(Pipeline):


    def run(self):
        mlflow.set_experiment(self.config.logger.experiment_hpo)
        hpo_loggers = {
            "local":loggers.LocalLogger(),
            "mlflow":loggers.MLFlowLogger(),
            "optuna":loggers.OptunaLogger(),
            }
        self.logger=PipelineLogger(loggers = hpo_loggers, logger_config = self.config.logger)

        self.study = self.setup_study()


        pruner = optuna.pruners.MedianPruner(n_startup_trials=config.hpo.warmup_trials,n_warmup_steps=config.hpo.warmup_steps)
        with mlflow.start_run() as run:
            self.parent_run = run.info.run_id
            self.study.optimize(self.objective, n_trials=config.hpo.trials)
            
            
        
        return self.get_optimal_parameters()
        
    def setup_study(self):
        #For the sake of the demo we limit ourselves to one active study.
        storage = MlflowStorage(experiment_id = mlflow.get_experiment_by_name(self.config.logger.experiment_hpo).experiment_id)
        try:
            optuna.delete_study(study_name=self.config.logger.study_name)
        except:
            pass
        study = optuna.create_study(study_name = self.config.logger.study_name , direction="minimize", storage = storage)
        return(study)
        
    def objective(self, trial):
        self.config.model.n_width = trial.suggest_int("n_width", 4,64)
        self.config.model.n_depth = trial.suggest_int("n_depth",0,4)
 
        with mlflow.start_run(run_name = f"Trial_{trial.number}", nested = True):
            runinfo = RunInfo(
                run_id = mlflow.active_run().info.run_id,
                parent_id = self.parent_run,
                trial = trial,
                study = trial.study
            )
            self.logger.update_runinfo(runinfo)
                              
            metrics = self.run_instance()

        trial.set_user_attr("configuration", self.config.dict())
        return metrics.test_loss
    
    def get_optimal_parameters(self):
        assert self.study is not None, "Run HPO or load an existing study first!"
        config= PipelineConfig.model_validate(self.study.best_trial.user_attrs["configuration"])
        return(config)

    def save(model, metrics, parameters = None):
        pass



class Pipeline_Retrain(Pipeline):


    def run(self):
        mlflow.set_experiment(self.config.logger.experiment_root)
        final_loggers = {
            "local":loggers.LocalLogger(),
            "mlflow":loggers.FinalLogger(),
            }
        self.logger=PipelineLogger(loggers = final_loggers, logger_config = self.config.logger)
        with mlflow.start_run() as run:
            runinfo = RunInfo(
                run_id = mlflow.active_run().info.run_id,
                parent_id = None,
                trial = None,
                study = None
            )
            self.logger.update_runinfo(runinfo)
            self.run_instance()

        #self.compare()
        #self.deploy()
    

    def save(model, metrics, parameters = None):
        #Create a tempdir and save parameters, model source, configuration json.
        pass



class Pipeline_Evaluator(Pipeline):
    def run(self):
        eval_loggers = {
            "local":loggers.LocalLogger()
            }
        self.logger=PipelineLogger(loggers = eval_loggers, logger_config = self.config.logger)

        #Evaluates the current champion and determines if it needs to be retrained.
        self.outcome = {
            "valid" : 0,
            "retrain": 1,
            "missing study": 2,

        }
        self.validate_experiment()
        study_exists = self.validate_study()

        if study_exists == False:
            return self.outcome["missing study"]
        
        self.model = self.get_alias("champion")
        
        if self.model is None:
            return self.outcome["retrain"]
        

        metrics = self.evaluate()


        #Logged metrics: 
        client = MlflowClient()
        run = client.get_run(run_id)
        old_metrics = run.data.metrics
        if metrics.loss > 2 * old_metrics["test_loss"][-1]:
            return self.outcome["retrain"]
        else:
            return self.outcome["valid"]
        


    def validate_experiment(self):
        client = MlflowClient()
        
        try:
            training = client.get_experiment_by_name(self.config.logger.experiment_train)
            if training is None:
                mlflow.create_experiment(self.config.logger.experiment_train)
                self.logger.log_message(f"Experiment {self.config.logger.experiment_train} does not exist yet, creating...")

        except MlflowException as e:
            mlflow.create_experiment(self.config.logger.experiment_train)
            self.logger.log_message(f"Experiment {self.config.logger.experiment_train} does not exist yet, creating...")
            
        
        try:
            hpo = client.get_experiment_by_name(self.config.logger.experiment_hpo)
            if hpo is None:
                mlflow.create_experiment(self.config.logger.experiment_hpo)
                self.logger.log_message(f"Experiment {self.config.logger.experiment_hpo} does not exist yet, creating...")
        except MlflowException as e:
            mlflow.create_experiment(self.config.logger.experiment_hpo)
            self.logger.log_message(f"Experiment {self.config.logger.experiment_hpo} does not exist yet, creating...")
            

    def validate_study(self):
        exp_id = mlflow.get_experiment_by_name(self.config.logger.experiment_hpo).experiment_id
        try:
            self.study = optuna.load_study(study_name = self.config.logger.study_name, storage = MlflowStorage(exp_id))
            return True
        except RestException as e:
            self.logger.log_message("Study does not exist yet.")
            return False
        except Exception as e:
            self.logger.log_message("Study does not exist yet.")
            print(e)
            return False




    def get_run(self, runid):
        #Fetch models from experiment.
        pass

    def get_alias(self, alias):
        client = MlflowClient()
        try:
            client.get_model_version_by_alias(self.config.logger.model_name, alias)
        except MlflowException as e:
            self.logger.log_message(f"No model with alias {alias} found.")
            return None
        



class Promotion_Pipeline(Pipeline_Evaluator):
    def run(self):
        #Evaluates the current champion and determines if it needs to be retrained.
        self.model = self.get_champion()
        metrics = self.evaluate()


    def compete(self):
        #Pitch a challenger against the current champion and promote it if it wins.
        pass




