import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath

from kfp import kubernetes
from typing import Optional


@dsl.container_component
def deploy_model():
    #aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    return dsl.ContainerSpec(
        "quay.io/redhat-ai-dev/utils:latest",
        ["/bin/sh", "-c"],
        [
            f"env && oc process -f https://raw.githubusercontent.com/sauagarwa/fraud-detection/main/deployment/inference-server-deployment.yaml -p INFERENCE_SERVER_NAME=fraud-detection-is -p AWS_ACCESS_KEY_ID={{AWS_ACCESS_KEY_ID}} -p AWS_SECRET_ACCESS_KEY={{AWS_SECRET_ACCESS_KEY}} -p AWS_S3_ENDPOINT={{AWS_S3_ENDPOINT}} -p AWS_DEFAULT_REGION={{AWS_DEFAULT_REGION}} -p AWS_S3_BUCKET={{AWS_S3_BUCKET}} | oc create -f -"
        ],
    )


@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline():

    deploy_model_task = deploy_model()

    kubernetes.use_secret_as_env(
        task=deploy_model_task,
        secret_name='aws-connection-my-storage',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })

if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )