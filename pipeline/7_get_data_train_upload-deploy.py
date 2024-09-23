import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath

from kfp import kubernetes
from typing import Optional


@dsl.component(base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301")
def get_data(data_version: str,
             bucket_name: str,
             data_output_path: OutputPath()):
    import urllib.request
    import os

    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')

    print("data version = " + data_version)
    print("endpoint url = " + endpoint_url)
    print("bucket = " + bucket_name)

    url = f"{endpoint_url}/{bucket_name}/{data_version}/card_transdata.csv"

    print("starting download...")
    urllib.request.urlretrieve(url, data_output_path)
    print("done")

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["tf2onnx", "seaborn"],
)
def train_model(data_input_path: InputPath(), model_output_path: OutputPath()):
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight
    import tf2onnx
    import onnx
    import pickle
    from pathlib import Path

    # Load the CSV data which we will use to train the model.
    # It contains the following fields:
    #   distancefromhome - The distance from home where the transaction happened.
    #   distancefromlast_transaction - The distance from last transaction happened.
    #   ratiotomedianpurchaseprice - Ratio of purchased price compared to median purchase price.
    #   repeat_retailer - If it's from a retailer that already has been purchased from before.
    #   used_chip - If the (credit card) chip was used.
    #   usedpinnumber - If the PIN number was used.
    #   online_order - If it was an online order.
    #   fraud - If the transaction is fraudulent.
    Data = pd.read_csv(data_input_path)

    # Set the input (X) and output (Y) data.
    # The only output data we have is if it's fraudulent or not, and all other fields go as inputs to the model.

    X = Data.drop(columns = ['repeat_retailer','distance_from_home', 'fraud'])
    y = Data['fraud']

    # Split the data into training and testing sets so we have something to test the trained model with.

    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle = False)

    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, stratify = y_train)

    # Scale the data to remove mean and have unit variance. This means that the data will be between -1 and 1, which makes it a lot easier for the model to learn than random potentially large values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train.values)

    Path("artifact").mkdir(parents=True, exist_ok=True)
    with open("artifact/test_data.pkl", "wb") as handle:
        pickle.dump((X_test, y_test), handle)
    with open("artifact/scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), we set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.

    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
    class_weights = {i : class_weights[i] for i in range(len(class_weights))}


    # Build the model, the model we build here is a simple fully connected deep neural network, containing 3 hidden layers and one output layer.

    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = len(X.columns)))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()


    # Train the model and get performance

    epochs = 2
    history = model.fit(X_train, y_train, epochs=epochs, \
                        validation_data=(scaler.transform(X_val.values),y_val), \
                        verbose = True, class_weight = class_weights)

    # Save the model as ONNX for easy use of ModelMesh

    model_proto, _ = tf2onnx.convert.from_keras(model)
    print(model_output_path)
    onnx.save(model_proto, model_output_path)


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(input_model_path: InputPath(), version_file_output_path: OutputPath()):

    # Save the model as ONNX for easy use of ModelMesh
    import os
    import boto3
    import botocore
    from datetime import datetime

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    model_version = datetime.now().strftime("%y-%m-%d-%H%M%S")
    print("model version = " + model_version)

    object_name = 'model-' + model_version +'/fraud/1/model.onnx'
    print("object name = " + object_name)

    # Set up the S3 client
    s3 = boto3.client('s3',
                      endpoint_url=endpoint_url,
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)


    # upload the model to the registry
    s3.upload_file(input_model_path, bucket_name, object_name)

    print("version output path = " + version_file_output_path)
    # store the model version in a file temporarily for the next pipeline job
    with open(version_file_output_path, "w") as text_file:
        text_file.write('model_version='+model_version)


# @dsl.container_component
# def deploy_model():
#     return dsl.ContainerSpec(
#         "quay.io/redhat-ai-dev/utils:latest",
#         ["/bin/sh", "-c"],
#         [
#             """oc delete secret aws-connection-fraud-detection-is || true && 
#             oc delete servingruntimes fraud-detection-is || true && 
#             oc delete inferenceservices fraud-detection-is || true && 
#             oc process -f https://raw.githubusercontent.com/sauagarwa/fraud-detection/main/deployment/inference-server-deployment.yaml \
#             -p INFERENCE_SERVER_NAME=fraud-detection-is \
#             -p AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
#             -p AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
#             -p AWS_S3_ENDPOINT=${AWS_S3_ENDPOINT} \
#             -p AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
#             -p AWS_S3_BUCKET=${AWS_S3_BUCKET} | oc create -f -"""
#         ],
#     )

@dsl.component(base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301")
def deploy_model(data_connection_name: str,
                 input_version_file_path: InputPath()):
    import os, time, subprocess
    from jinja2 import Template
    import urllib.request
    import os

    variables = {}
    with open(input_version_file_path) as myfile:
        for line in myfile:
            name, var = line.partition("=")[::2]
            variables[name.strip()] = str(var)

    template_data = {"model_version": variables['model_version'], "storage_key": data_connection_name}

    def download_file(dir, filename):
        url = f"https://raw.githubusercontent.com/sauagarwa/fraud-detection/main/deployment/{filename}"

        print("starting download...")
        urllib.request.urlretrieve(url, f"{dir}/{filename}")
        print("done")

        pass

    def deploy_template(filename, template_data):
        print("invoking template:" + filename)
        template = Template(open(filename).read())
        rendered_template = template.render(template_data)
        
        subprocess.run(['oc', 'whoami'])
        ps = subprocess.Popen(['echo', rendered_template], stdout=subprocess.PIPE)
        print(ps.stdout)
        output = subprocess.check_output(['oc', 'apply', '-f', '-'], stdin=ps.stdout)
        ps.wait()
        print(f"Deployed template {filename}. Version: {variables['model_version']}")
        
    templates = ['serving-runtime.yaml',
                 'inference-service.yaml']
    for t in templates:
        dir =  "/tmp"
        download_file(dir,t)
        deploy_template(f"{dir}/{t}", template_data)


@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(data_version: str = '1',
             bucket_name: str = 'raw-data'):
    
    secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
    }
    get_data_task = get_data(data_version=data_version, bucket_name=bucket_name)
    get_data_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        task=get_data_task,
        secret_name="aws-connection-fraud-detection",
        secret_key_to_env=secret_key_to_env)
    csv_file = get_data_task.outputs["data_output_path"]
    # csv_file = get_data_task.output
    train_model_task = train_model(data_input_path=csv_file)
    train_model_task.set_caching_options(False)
    onnx_file = train_model_task.outputs["model_output_path"]

    upload_model_task = upload_model(input_model_path=onnx_file)
    upload_model_task.set_caching_options(False)
    #upload_model_task.set_env_variable(name="S3_KEY", value="models/fraud/1/model.onnx")
    kubernetes.use_secret_as_env(
        task=upload_model_task,
        secret_name="aws-connection-fraud-detection",
        secret_key_to_env=secret_key_to_env)
    version_file = upload_model_task.outputs["version_file_output_path"]

    deploy_model_task = deploy_model(data_connection_name="aws-connection-fraud-detection",
                                     input_version_file_path=version_file).after(upload_model_task)
    deploy_model_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        task=deploy_model_task,
        secret_name="aws-connection-fraud-detection",
        secret_key_to_env=secret_key_to_env)


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )
