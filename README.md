
An example that demonstrates how to call Tensorflow Serving API via gRPC protocol using Golang.


# Steps

## Build SavedModel

To build SavedModel, follow instructions on https://medium.freecodecamp.org/how-to-deploy-tensorflow-models-to-production-using-tf-serving-4b4b78d41700

## Run TensorFlow Serving Service

Put `run_docker.sh` besides the generated `versions` folder, then run command:

```
./run_docker.sh
```

Or

```

docker run --detach --rm \
  --name deeplab_v3 \
  -p 8500:8500 -p 8501:8501 \
  -v "$(pwd)/versions:/models/deeplab_v3" \
  -e MODEL_NAME=deeplab_v3 \
  -e TF_CPP_MIN_VLOG_LEVEL=1 \
  tensorflow/serving:latest

```

## Build Golang Client

Put this project in `$GOPATH/src/your-project-name`
Build client:

```
dep ensure -v -update
./build_proto.sh
mkdir -p ./bin && go build -o ./bin/client
```

Protos under `proto_files` are collected from `https://github.com/tensorflow/serving/tree/r1.13` and `https://github.com/tensorflow/tensorflow/tree/r1.13`

`go_package` path are modified to make generated `.pb.go` files work with `vendor` folder.


## Run Client

```
./bin/client test-01.png
```

## View Result
Output files:

```
segmap.json
segmap.png
```
