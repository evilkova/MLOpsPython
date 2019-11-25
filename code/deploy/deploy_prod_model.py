from azureml.core.model import InferenceConfig
from azureml.core.webservice import AksWebservice
from azureml.core import Run, Model
import argparse
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath("./ml_service/util"))  # NOQA: E402
from workspace import get_workspace
from attach_compute import get_compute

load_dotenv()
workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
resource_group = os.environ.get("RESOURCE_GROUP")
subscription_id = os.environ.get("SUBSCRIPTION_ID")
tenant_id = os.environ.get("TENANT_ID")
app_id = os.environ.get("SP_APP_ID")
app_secret = os.environ.get("SP_APP_SECRET")
deploy_script_path = os.environ.get("DEPLOY_PROD_SCRIPT_PATH")
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")
compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
model_name = os.environ.get("MODEL_NAME")
build_id = os.environ.get("BUILD_BUILDID")
pipeline_name = os.environ.get("DEPLOY_PROD_PIPELINE_NAME")
service_name = os.environ.get("DEPLOY_PROD_SERVICE_NAME")
sources_directory_train = os.environ.get("SOURCES_DIR_TRAIN")

# Get Azure machine learning workspace
aml_workspace = get_workspace(
    workspace_name,
    resource_group,
    subscription_id,
    tenant_id,
    app_id,
    app_secret)
print(aml_workspace)

# Get Azure machine learning cluster
aml_compute = get_compute(
    aml_workspace,
    compute_name,
    vm_size)
if aml_compute is not None:
    print(aml_compute)

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
    default="prod-abn-ca-service"
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

deployment_config = AksWebservice.deploy_configuration(cpu_cores=1,
                                                       memory_gb=4)
service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       deployment_target=aml_compute,
                       overwrite=True)
service.wait_for_deployment(show_output=True)
