from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline  # , PipelineData
from azureml.core.runconfig import RunConfiguration, CondaDependencies
# from azureml.core import Datastore
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath("./ml_service/util"))  # NOQA: E402
from workspace import get_workspace
from attach_compute import get_compute


def main():
    load_dotenv()
    workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    deploy_script_path = os.environ.get("DEPLOY_SCRIPT_PATH")
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    model_name = os.environ.get("MODEL_NAME")
    build_id = os.environ.get("BUILD_BUILDID")
    pipeline_name = os.environ.get("DEPLOY_PIPELINE_NAME")
    service_name = os.environ.get("DEPLOY_SERVICE_NAME")
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

    run_config = RunConfiguration(conda_dependencies=CondaDependencies.create(
        conda_packages=['numpy', 'pandas',
                        'scikit-learn'],
        pip_packages=['azure', 'azureml-core',
                      'azure-storage',
                      'azure-storage-blob'])
    )
    run_config.environment.docker.enabled = True

    model_name = PipelineParameter(
        name="model_name", default_value=model_name
    )
    release_id = PipelineParameter(
        name="release_id", default_value="0"
    )
    service_name = PipelineParameter(
        name="service_name", default_value=service_name
    )

    deploy_step = PythonScriptStep(
        name="Deploy Model",
        script_name=deploy_script_path,
        compute_target=aml_compute,
        source_directory=sources_directory_train,
        arguments=[
            "--release_id", release_id,
            "--model_name", model_name,
            "--service_name", service_name
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Deploy created")

    steps = [deploy_step]

    deploy_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    deploy_pipeline.validate()
    published_pipeline = deploy_pipeline.publish(
        name=pipeline_name,
        description="Model deploy  pipeline",
        version=build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()
