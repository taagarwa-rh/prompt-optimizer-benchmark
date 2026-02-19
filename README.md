# Benchmark



## Prerequisites

1. Clone the repo

1. Move to this directory

    ```sh
    cd prompt-optimization-benchmark
    ```

2. Copy `example_config.yaml` to `config.yaml` and update the config

    ```sh
    cp example_config.yaml config.yaml
    ```

## Run locally

1. Start up the MLflow server:

    ```sh
    uv run mlflow server
    ```

2. Run:

    ```sh
    uv run benchmark.py --config config.yaml
    ```


## Run in Openshift

1. Set your namespace, MLFlow tracking URI and experiment name:
   
    ```sh
    export NAMESPACE="prompt-optimization-benchmark"
    export MLFLOW_TRACKING_URI="your-tracking-uri"
    ```

1. Create a ConfigMap with your configuration:

    ```sh
    oc create configmap -n $NAMESPACE prompt-optimization-benchmark-config --from-file=config.yaml
    ```

1. Run the benchmark:

    ```sh
    oc process -f benchmark.yaml \
        -p NAMESPACE="$NAMESPACE" \
        -p MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
        | oc apply -f -
    ```