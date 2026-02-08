"""
Welcome to the first example!

Implement the TODO's listed in this file to make the training loop (that you can initially run by executing this file)
part of an optuna optimization study.

"""
from configs.model_config import ModelConfig
from demo.training import Pipeline
from demo.loggers import (
LocalLogger,
OptunaLogger,
MLFlowLogger,
FinalLogger
)
from demo.constants import LOGGERS, OPTUNA_STUDY_NAME,MLFLOW_EXPERIMENT_NAME
from demo.util import load_model_config, load_mlflow_config, load_optuna_config


from demo.loggers import (
    PipelineLogger,
)



import mlflow
import optuna


class OptunaStudyRunner:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.study = None
        self.config = load_optuna_config()
        self.config.study.study = OPTUNA_STUDY_NAME

        #Set up a pruner (median pruner):
        pruner = optuna.pruners.MedianPruner(n_startup_trials=self.config.n_startup_trials, n_warmup_steps=self.config.n_warmup_steps)

        #Define our sampler (TPE is by default, but we also set the seed here):
        sampler = optuna.samplers.TPESampler(seed=self.config.study.seed)

        self.study = optuna.create_study(
            study_name=self.config.study.study,
            direction="minimize",
            pruner=pruner,
            storage=self.config.storage.uri,
            sampler=sampler,
            load_if_exists=self.config.study.load_if_exists
        )

    def objective(self, trial):
        config=self.pipeline.config
        runinfo=self.pipeline.runinfo
        logger=self.pipeline.logger


        #Initialize our logger:
        optuna_logger=OptunaLogger(runinfo)
        logger.set_logger(key=LOGGERS.OPTUNA.value, logger=optuna_logger)

        #Add the current trial to the runinfo of the model
        runinfo.trial = trial

        #Let us try optimize a few parameters, for example the width and the depth of the network:
        config.n_width = trial.suggest_int("width", 4, 32)
        config.n_depth = trial.suggest_int("depth", 0, 3) #At 0 depth we have more of a linear regressor than an mlp.
        #Now we may run the pipeline as is, or we may wrap it up in mlflow first:
        mlflowdriver(self.pipeline)
        #self.pipeline.run()
        return self.pipeline.test_metrics.test_loss
    def _optimize(self):


        # See: configs/optuna.yaml:
        config = self.config

        self.study.optimize(self.objective, n_trials=config.n_trials, timeout=config.timeout)

    def finalize(self):
        self.pipeline.config=ModelConfig.model_validate(self.study.best_trial.user_attrs["config"])
        self.pipeline.logger.reset_logger()


        mlflowdriver(self.pipeline, FinalLogger)

        return self.pipeline.test_metrics.test_loss

def mlflowdriver(pipeline: Pipeline, Logger : MLFlowLogger = MLFlowLogger):
    mlflow_config = load_mlflow_config()
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


    with mlflow.start_run() as run:
        runinfo=pipeline.runinfo

        runinfo.run_id = run.info.run_id
        mlflow_logger=Logger(runinfo)

        pipeline.logger.set_logger(key=LOGGERS.MLFLOW.value, logger=mlflow_logger)
        pipeline.run()

    return pipeline




def run_project():
    """
    Load model config, set up loggers and creater the pipeline.

    This pipeline can be run as is, but we will wrap it in optuna and mlflow.
    The pipeline logger is a logger class that collects and passes on logging calls.

    You may use the constant keys under the constant enumerator LOGGERS to get the keys for:

    -LOCAL
    -OPTUNA
    -MLFLOW
    -FINAL
    """
    config = load_model_config()
    local_logger = LocalLogger(show_figure=False)

    pipelinelogger = PipelineLogger()
    pipelinelogger.set_logger(key=LOGGERS.LOCAL.value, logger=local_logger)

    pipeline = Pipeline(config=config, logger=pipelinelogger)
    opt_runner = OptunaStudyRunner(pipeline)
    opt_runner._optimize()
    opt_runner.finalize()


run_project()
