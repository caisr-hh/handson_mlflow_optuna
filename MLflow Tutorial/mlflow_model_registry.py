from models.sample_model import MlpModel
from configs.pipeline_config import DataConfig, ModelConfig

from data.data import EpochMetrics, TestMetrics, construct_data
import torch
import mlflow
from dataclasses import asdict
from plots import plot_res




class Pipeline():

    def __init__(self):

        dataconfig = DataConfig()
        self.study = None
        self.data = construct_data(dataconfig)
        self.n_width = 32
        self.n_depth = 3

    def run(self):

        ############################################################################################################
        # Set the tracking server (make sure to run mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Create or set experiment:
        mlflow.set_experiment("ecml_local_registering_example")
        # Start an mlflow run
        with mlflow.start_run():
            self.run_instance()
        ###########################################################################################################




    def run_instance(self):

        config = ModelConfig(
            n_width = self.n_width,
            n_depth = self.n_depth
        )


        #Initiate model based on the suggested configuration
        self.model = MlpModel(config)

        #Log our parameters:
        mlflow.log_params(config.model_dump())

        # We log the training loss inside the loop.
        self.train()

        # Run evaluation pipeline and get metrics
        metrics = self.evaluate()

        #And here we log the test...
        mlflow.log_metrics(asdict(metrics))

        #Let us set a tag in case we need additional identification:
        mlflow.set_tags({
            "model": "mlp",
            "project": "ecml",
            "data": "circles"
        })


        # Let us log a figure of our results.
        figure = plot_res(self.model,self.data)
        mlflow.log_figure(figure, "Boundary.png")

        input_example = self.data.test_loader.dataset[0:10][0]
        with torch.no_grad():
            output_example = self.model(input_example)
        signature = mlflow.models.infer_signature(input_example.numpy(), output_example.numpy())

        # Log the model as an onnx model

        onnx_program = torch.onnx.export(self.model,
                                         input_example,
                                         f=None,
                                         dynamic_shapes={"in_tensor": {0: "batch_size"}}
                                         )

        onnx_model = onnx_program.model_proto

        #Reigister the model under the onnx flavour, producing all artefacts necessary to host the model
        mlflow.onnx.log_model(onnx_model, registered_model_name="registered_model", signature=signature)


        return metrics.test_loss




    def train(self) -> MlpModel:
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

            # We can log dictionaries of data, so dumping pydantic basemodels or dataclasses.
            mlflow.log_metrics(asdict(metrics), epoch)

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