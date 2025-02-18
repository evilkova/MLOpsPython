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
    sources_directory_train = os.environ.get("SOURCES_DIR_TRAIN")
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH")
    evaluate_script_path = os.environ.get("EVALUATE_SCRIPT_PATH")
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_CPU_SKU")
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    model_name = os.environ.get("MODEL_NAME")
    build_id = os.environ.get("BUILD_BUILDID")
    pipeline_name = os.environ.get("TRAINING_PIPELINE_NAME")

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

    conda_dependencies = CondaDependencies.create(
        conda_packages=[
            'numpy',
            'pandas',
            'scikit-learn'
        ],
        pip_packages=[
            'azureml-core==1.0.72.*',
            'azureml-sdk==1.0.72.*',
            'azure-storage',
            'azure-storage-blob',
            'azureml-dataprep',
            'azureml-datadrift==1.0.72.*'
        ],
        pin_sdk_version=False
    )

    print(conda_dependencies.serialize_to_string())

    run_config = RunConfiguration(
        framework='Python',
        conda_dependencies=conda_dependencies
    )
    run_config.environment.docker.enabled = True

    model_name = PipelineParameter(
        name="model_name", default_value=model_name)
    release_id = PipelineParameter(
        name="release_id", default_value="0"
    )

    train_step = PythonScriptStep(
        name="Train Model",
        script_name=train_script_path,
        compute_target=aml_compute,
        source_directory=sources_directory_train,
        arguments=[
            "--release_id", release_id,
            "--model_name", model_name,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Train created")

    evaluate_step = PythonScriptStep(
        name="Evaluate Model ",
        script_name=evaluate_script_path,
        compute_target=aml_compute,
        source_directory=sources_directory_train,
        arguments=[
            "--release_id", release_id,
            "--model_name", model_name,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    evaluate_step.run_after(train_step)
    steps = [evaluate_step]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="Model training/retraining pipeline",
        version=build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')


if __name__ == '__main__':
    main()
