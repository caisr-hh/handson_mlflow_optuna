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

        # TODO: Define our sampler (TPE is by default, but we also set the seed here with the config):
        # sampler = optuna.samplers.TPESampler(seed=..)

        # TODO: Create our study:
        self.study = optuna.create_study(
            study_name=self.config.study.study,
            # TODO: direction=,
            storage=self.config.storage.uri,
            # TODO: sampler=sampler,
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

        # TODO: Suggest "width" between 4 and 32 and "depth" between 0 and 3:
        # config.n_width = trial.suggest_...
        # config.n_depth = trial.suggest...
        # TODO: run the pipeline:

        # TODO: We can get the test loss by self.pipeline.test_metrics.test_loss, return it for optimization!
        # return ...

    def _optimize(self):

        # See: configs/optuna.yaml:
        config = self.config

        # TODO: Add our objective function, and add the number of trials from our config
        # self.study.optimize(func=, n_trials=, timeout=config.timeout)


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
    local_logger = LocalLogger(
        show_figure=False
    )  # TODO: When running the pipeline naked you may set this to true.

    pipelinelogger = PipelineLogger()
    pipelinelogger.set_logger(key=LOGGERS.LOCAL.value, logger=local_logger)

    pipeline = Pipeline(config=config, logger=pipelinelogger)

    # TODO:
    # opt_runner = OptunaStudyRunner(pipeline)

    pipeline.run()  # <---- replace with: opt_runner._optimize()


run_project()
