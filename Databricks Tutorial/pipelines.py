
#Tutorial repo---------------------------------
from configs.pipeline_config import PipelineConfig, RunInfo
from models.sample_model import MlpModel
from data.data import EpochMetrics, TestMetrics, get_data_db
from logs.logger import PipelineLogger, Logger
import logs.logger as loggers
from misc.util import load_pipeline_config
from misc.exceptions import HaltTraining
from misc.plots import plot_res
#----------------------------------------------

# MLFLOW--------------------------------------
import mlflow
import mlflow.deployments
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
        self.logger = PipelineLogger(loggers = {}, logger_config = config.logger) #Empty logger by default
        self.study = None
        self.data = get_data_db(self.config.data)
        self.config.data.version = self.data.config.version #From flexibly resolving the data version to a fixed version
    def run(self):

        local_loggers = {
            "local":loggers.LocalLogger(),
            }
        self.logger=PipelineLogger(loggers = local_loggers, logger_config = self.config.logger)
        with mlflow.start_run(nested = False) as run:
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
            fig = plot_res(self.model,self.data)
            self.save(self.model)
            self.logger.log_figure(fig)
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
        try:
            self.model.eval()
        except NotImplementedError: #Hotfix for export in graph format without eval()...
            pass
        
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
    

    def save(self, model):
        pass

    def load_study(self):
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
            
    def get_optimal_parameters(self):
        #We can get the best trial (frozen) from omptuna and load parameters from there
        assert self.study is not None, "Run HPO or load an existing study first!"
        trial = self.study.best_trial
        self.config.model.n_width = trial.params["n_width"]
        self.config.model.n_depth = trial.params["n_depth"]

        return(self.config.model)


class Pipeline_HPO(Pipeline):


    def run(self):
        mlflow.set_experiment(self.config.logger.experiment_hpo)
        hpo_loggers = {
            "local":loggers.LocalLogger(),
            "optuna":loggers.OptunaLogger(),
            }
        self.logger=PipelineLogger(loggers = hpo_loggers, logger_config = self.config.logger)

        self.study = self.setup_study()


        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.config.hpo.warmup_trials,n_warmup_steps=self.config.hpo.warmup_steps)

        self.study.optimize(self.objective, n_trials=self.config.hpo.trials)
            
            
        
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

        run_id = trial._trial_id #Note that the mlflow backend means the trial id for optuna is the same as mlflows run id for that trial.


        runinfo = RunInfo(
            run_id = run_id,
            parent_id = None,
            trial = trial,
            study = trial.study
        )
        self.logger.update_runinfo(runinfo)
                            
        metrics = self.run_instance()
   

        trial.set_user_attr("configuration", self.config.dict())

        return metrics.test_loss
    


    def save(model, metrics, parameters = None):
        pass



class Pipeline_Retrain(Pipeline):


    def run(self):
        mlflow.set_experiment(self.config.logger.experiment_train)
        final_loggers = {
            "local":loggers.LocalLogger(),
            "mlflow":loggers.FinalLogger(),
            }
        self.logger=PipelineLogger(loggers = final_loggers, logger_config = self.config.logger)
        assert self.load_study(), "Please run HPO first!"
        self.config.model = self.get_optimal_parameters()

        with mlflow.start_run() as run:
            runinfo = RunInfo(
                run_id = mlflow.active_run().info.run_id,
                parent_id = None,
                trial = None,
                study = None
            )
            self.logger.update_runinfo(runinfo)
            self.run_instance()

    

    def save(self, model):
        self.logger.log_model(model)




class Pipeline_Evaluator(Pipeline):
    def run(self):

        eval_loggers = {
            "local":loggers.LocalLogger(),
            }
        self.logger=PipelineLogger(loggers = eval_loggers, logger_config = self.config.logger)

        #Evaluates the current champion and determines if it needs to be retrained.
        self.outcome = {
            "valid" : 0,
            "retrain": 1,
            "missing study": 2,

        }
        self.validate_experiment()
        study_exists = self.load_study()
        if study_exists == False:
            return self.outcome["missing study"]
        
        #Gets models and version infos from registered model aliases!
        self.model = self.get_alias_model("champion")
        self.info = self.get_alias_version("champion")

        #Study exists but no champion, interrupted pipeline, retrain:
        if self.model is None:
            return self.outcome["retrain"]


        

        

        metrics = self.evaluate()


        #Logged metrics: 
        client = MlflowClient()
        run_id = self.info.run_id
        run = client.get_run(run_id)
        old_metrics = run.data.metrics
 
        if metrics.test_loss > self.config.retrain_threshold * old_metrics["test_loss"]: # todo: add constant to config
            return self.outcome["retrain"]
        else:
            return self.outcome["valid"]
        
    def get_alias_model(self, alias):
        try:
            model = mlflow.pytorch.load_model(model_uri=f"models:/{self.config.logger.model_name_uc}@{alias}")
            
            return model
        except MlflowException as e:
            self.logger.log_message(f"No model with alias {alias} found.")
            return None

    def get_alias_version(self, alias):
        client = MlflowClient()
        try:
            info = client.get_model_version_by_alias(self.config.logger.model_name_uc, alias)

            return info
        except MlflowException as e:
            self.logger.log_message(f"No model with alias {alias} found.")
            return None

    def get_run(self, runid):
        #Fetch models from experiment.
        pass




class Pipeline_Promote(Pipeline_Evaluator):
    def run(self):
        #Evaluates the current champion and determines if it needs to be retrained.
        update = self.compete()
        if update:
            self.promote()
        
        return update

    def compete(self) -> bool:
        #Compares the current champion and the condender, returning True if the contender should be promoted and deployed.
        self.model = self.get_alias_model("champion")
        if self.model != None:
            #Get and evaluate model on new data

            metrics_old = self.evaluate()
            self.model = self.get_alias_model("contender")
            assert self.model is not None, "Contender not found, nothing to promote!"
            metrics_new = self.evaluate()
            if metrics_new.test_loss < metrics_old.test_loss:
                return(True)
            else:
                self.logger.log_message("Contender did not beat the champion, keeping champion.")
                return(False)
        else:

            self.model = self.get_alias_model("contender")
            assert self.model is not None, "Contender not found, nothing to promote."
            return(True) #No existing champion, victory by default.
    def promote(self):
        #Promote the contender to champion
        client = MlflowClient()
        info = self.get_alias_version("contender")
        client.set_registered_model_alias(name=info.name, alias="champion", version=info.version)
        client.delete_registered_model_alias(name=info.name, alias="contender")
        self.logger.log_message(f"Promoted model {info.name} version {info.version} to champion.")

class Pipeline_Deploy(Pipeline_Evaluator):
    def run(self):
        #Evaluates the current champion and determines if it needs to be retrained.
 
        self.deploy()
        
        return



    def deploy(self):
        #Deploy the model to a serving endpoint
        deploy_client = mlflow.deployments.get_deploy_client("databricks")
        endpoint = None
        info = self.get_alias_version("champion")
        config = {
            "served_entities":[
                {
                    "entity_name": self.config.logger.model_name_uc,
                    "entity_version": info.version,
                    "scale_to_zero_enabled":True,
                    "workload_size": "Small"

                }
            ]
        }
        try:
            endpoint = deploy_client.get_endpoint(self.config.logger.endpoint_name)

        except:
            self.logger.log_message("Unable to find existing endpoint, creating new...")

        if endpoint is None:
            self.logger.log_message("Creating new endpoint...")
            deploy_client.create_endpoint(self.config.logger.endpoint_name, config)
        else:
            self.logger.log_message("Updating existing endpoint")
            deploy_client.update_endpoint(self.config.logger.endpoint_name, config)
        self.logger.log_message(f"Deployed model {info.name} version {info.version} to endpoint {self.config.logger.endpoint_name}.")




