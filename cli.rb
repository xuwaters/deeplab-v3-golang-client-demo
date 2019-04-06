require 'thor'

class CLI < Thor
  include CLI::Actions

  def initialize(*args)
    super(*args)
    @curr_dir = File.expand_path('.', __dir__)
  end

  desc 'gen_pb', 'generate tensorflow proto files'
  def gen_pb
    inside(@curr_dir) do
      proto_files = 'proto_files'
      options = %(-I #{proto_files} --go_out=plugins=grpc:vendor)

      run %(protoc #{options} #{proto_files}/tensorflow/core/example/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow/core/framework/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow/core/lib/core/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow/core/protobuf/*.proto)

      run %(protoc #{options} #{proto_files}/tensorflow_serving/apis/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow_serving/config/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow_serving/core/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow_serving/sources/storage_path/*.proto)
      run %(protoc #{options} #{proto_files}/tensorflow_serving/util/*.proto)
    end
  end
end

CLI.start if caller.empty?
