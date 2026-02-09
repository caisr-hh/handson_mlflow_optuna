from demo.training import Pipeline
from demo.loggers import LocalLogger, OptunaLogger, MLFlowLogger, FinalLogger
from demo.constants import LOGGERS, OPTUNA_STUDY_NAME, MLFLOW_EXPERIMENT_NAME
from demo.util import load_model_config, load_mlflow_config, load_optuna_config


from demo.loggers import (
    PipelineLogger,
)


import mlflow
import optuna


class OptunaStudyRunner:
    def __init__(self, pipeline):
        """We initialize our study, setting the active pipeline to be used during optimization and
        loading the configuration for our study defined in /configs.


        """
        self.pipeline = pipeline
        self.study = None

        # See: configs/optuna.yaml:
        self.config = load_optuna_config()
        self.config.study.study = OPTUNA_STUDY_NAME

        # Set up a pruner (median pruner):
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.config.n_startup_trials,
            n_warmup_steps=self.config.n_warmup_steps,
        )

        # Define our sampler (TPE is by default, but we also set the seed here):
        sampler = optuna.samplers.TPESampler(seed=self.config.study.seed)

        # Create our study:
        self.study = optuna.create_study(
            study_name=self.config.study.study,
            direction="minimize",
            pruner=pruner,
            storage=self.config.storage.uri,
            sampler=sampler,
            load_if_exists=self.config.study.load_if_exists,
        )

    def objective(self, trial):
        """
        This objective is run for each new trial, a guess for new hyperparameter configurations.
        First we initialize the optuna logger that is responsible for handling calls relevant for optuna.
        Use trial.suggest_float, ..._int, ..._categorical with minimum or maximum. For documentation see:

        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html

        Once the configuration has been set up we can either run the pipeline and our study directly or we can add a mlflow layer
        by using the mlflow runner we are setting up here.


        """
        config = self.pipeline.config
        runinfo = self.pipeline.runinfo
        logger = self.pipeline.logger

        # Add the current trial to the runinfo of the model
        runinfo.trial = trial

        # Initialize our logger:
        optuna_logger = OptunaLogger(runinfo)
        logger.set_logger(key=LOGGERS.OPTUNA.value, logger=optuna_logger)

        # Let us try optimize a few parameters, for example the width and the depth of the network:
        config.n_width = trial.suggest_int("width", 4, 32)
        config.n_depth = trial.suggest_int("depth", 0, 3)

        # TODO:instead of running the pipeline as below, let us call mlflowdriver with the pipeline (and leave the logger as the default)
        self.pipeline.run()  # <--- Remove!
        # mlflowdriver(...)
        return self.pipeline.test_metrics.test_loss

    def _optimize(self):

        # See: configs/optuna.yaml:
        config = self.config

        self.study.optimize(
            self.objective, n_trials=config.n_trials, timeout=config.timeout
        )

    def finalize(self):

        return


def mlflowdriver(pipeline: Pipeline, Logger: MLFlowLogger = MLFlowLogger):
    mlflow_config = load_mlflow_config()
    # TODO:Set tracking URI (mlflow_config.tracking_uri)
    # mlflow.set_tracking_uri(...)
    # TODO: Set experiment by name (use the constant MLFLOW_EXPERIMENT_NAME)
    # mlflow.set_experiment(...)

    with mlflow.start_run() as run:
        runinfo = pipeline.runinfo

        # Gets the runid of the active run
        runinfo.run_id = run.info.run_id
        # Initialize logger
        mlflow_logger = Logger(runinfo)

        pipeline.logger.set_logger(key=LOGGERS.MLFLOW.value, logger=mlflow_logger)
        # Run pipeline!
        pipeline.run()

    return pipeline


def run_project():
    """
    Load model config, set up loggers and create the pipeline.

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


run_project()
