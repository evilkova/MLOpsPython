from azureml.core.model import InferenceConfig
from azureml.core.webservice.aci import AciServiceDeploymentConfiguration
from azureml.core import Run, Model
import argparse

parser = argparse.ArgumentParser("deploy")
parser.add_argument(
    "--release_id",
    type=str,
    help="The ID of the release triggering this pipeline run",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the Model",
    default="sklearn_abn_ca_model.pkl",
)
parser.add_argument(
    "--service_name",
    type=str,
    help="Name of the datastore",
    default="abn-ca-service"
)

args = parser.parse_args()

model_name = args.model_name
release_id = args.release_id
service_name = args.service_name

print(model_name)
print(release_id)
print(service_name)

# Get workspace
run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

# Get workspace
inference_config = InferenceConfig(runtime="python",
                                   entry_script="score.py",
                                   conda_file="conda_dependencies.yml",
                                   source_directory="./deploy/scoring/")
print(inference_config)

# Do something with imagecnfig image_config = ContainerImage

# Get latest model
model_list = Model.list(ws, name=model_name)
model = next(
    filter(
        lambda x: x.created_time == max(
            model.created_time for model in model_list),
        model_list,
    )
)

# to do: get config from file
deployment_config = AciServiceDeploymentConfiguration(cpu_cores=1,
                                                      memory_gb=4,
                                                      location='westeurope',
                                                      collect_model_data=True)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       deployment_target=None,
                       overwrite=True)

print(service.state)

service.wait_for_deployment(show_output=True)
