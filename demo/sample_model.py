import torch

from configs.model_config import RunInfo, ModelConfig
from torch import nn
import yaml
from demo.constants import CONFIG_DIR


class MlpModel(nn.Module):
    #Just a MLP that can be generalized down to a linear regression if depth=0.
    def __init__(self, config: ModelConfig, runinfo: RunInfo | None = None):
        super(MlpModel, self).__init__()
        if config.n_depth > 0:
            inp_layer = nn.Sequential(
                nn.Linear(in_features=2, out_features=config.n_width), nn.ReLU()
            )
            intermediates = [
                nn.Sequential(
                    nn.Linear(in_features=config.n_width, out_features=config.n_width),
                    nn.ReLU(),
                )
                for i in range(config.n_depth - 1)
            ]
            out_layer = nn.Sequential(
                nn.Linear(in_features=config.n_width, out_features=1), nn.Sigmoid()
            )

            self.layers = nn.Sequential(inp_layer, *intermediates, out_layer)

        else:
            self.layers = nn.Sequential(
                nn.Linear(in_features=2, out_features=1), nn.Sigmoid()
            )

        self.config = config
        self.runinfo = runinfo

    def forward(self, in_tensor: torch.Tensor):

        return self.layers(in_tensor)
