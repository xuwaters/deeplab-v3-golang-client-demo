#!/usr/bin/env bash

curr_dir=$(cd "$(dirname "$0")"; pwd)

cd $curr_dir

proto_files="proto_files"
options="-I ${proto_files} --go_out=plugins=grpc:vendor"

protoc ${options} ${proto_files}/tensorflow/core/example/*.proto
protoc ${options} ${proto_files}/tensorflow/core/framework/*.proto
protoc ${options} ${proto_files}/tensorflow/core/lib/core/*.proto
protoc ${options} ${proto_files}/tensorflow/core/protobuf/*.proto

protoc ${options} ${proto_files}/tensorflow_serving/apis/*.proto
protoc ${options} ${proto_files}/tensorflow_serving/config/*.proto
protoc ${options} ${proto_files}/tensorflow_serving/core/*.proto
protoc ${options} ${proto_files}/tensorflow_serving/sources/storage_path/*.proto
protoc ${options} ${proto_files}/tensorflow_serving/util/*.proto
