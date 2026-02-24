# Prompt Optimization Benchmark

Run a benchmark to compare prompt optimization methods.

Leverages the [prompt-optimizer package](https://github.com/taagarwa-rh/prompt-optimizer).

## Prerequisites

1. Clone the repo

    ```sh
    git clone https://github.com/taagarwa-rh/prompt-optimizer-benchmark.git
    ```

2. Move to this directory

    ```sh
    cd prompt-optimization-benchmark
    ```

3. Copy `example_config.yaml` to `config.yaml` and update the config

    ```sh
    cp example_config.yaml config.yaml
    ```

## Run locally


1. Install the requirements:
   
    ```sh
    uv sync
    ```

2. Start up the MLflow server:

    ```sh
    uv run mlflow server
    ```

3. Run:

    ```sh
    uv run benchmark.py --config config.yaml
    ```


## Run in Openshift

1. Set your namespace and MLFlow tracking URI:
   
    ```sh
    export NAMESPACE="prompt-optimization-benchmark"
    export MLFLOW_TRACKING_URI="your-tracking-uri"
    ```

2. Create a ConfigMap with your configuration:

    ```sh
    oc create configmap -n $NAMESPACE prompt-optimization-benchmark-config --from-file=config.yaml
    ```

3. Run the benchmark:

    ```sh
    oc process -f benchmark.yaml \
        -p NAMESPACE="$NAMESPACE" \
        -p MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
        | oc apply -f -
    ```