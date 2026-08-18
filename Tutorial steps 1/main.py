from pipelines import TrainingPipeline
from util import load_pipeline_config
import logs.logger as loggers
from configs.pipeline_config import PipelineConfig



config = load_pipeline_config()

loggers = {
    "local":loggers.LocalLogger(config.logger)
    }

pipeline = TrainingPipeline(config, loggers)


pipeline.run()



