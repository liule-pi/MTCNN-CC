#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::Tensor;
using tensorflow::Status;

Status LoadGraph(const std::string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

int main()
{
  std::string root_dir = "";
  std::string img_file = "./data/test.jpg";
  std::string graph_file = "./data/ckpt/pnet/pnet_frozen.pb";
  std::string input_layer = "pnet/input";
  std::string output_layer_1 = "pnet/conv4-2/BiasAdd:0";
  std::string output_layer_2 = "pnet/prob1:0";
  // read img with opencv
  cv::Mat input_img = cv::imread(img_file);
  cv::Mat temp_img;

  cv::cvtColor(input_img, temp_img, cv::COLOR_BGR2RGB);
  temp_img = temp_img.t();

  // covert to tensor, normalization
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                  {1, temp_img.rows, temp_img.cols, 3}));
  float *ptr = input_tensor.flat<float>().data();
  cv::Mat tensor_img(temp_img.rows, temp_img.cols, CV_32FC3, ptr);
  temp_img.convertTo(tensor_img, CV_32FC3, 1/127.5, -1.0);


  std::unique_ptr<tensorflow::Session> session;
  /* std::string graph_path = tensorflow::io::JoinPath(root_dir, graph_file); */
  /* Status load_graph_status = LoadGraph(graph_path, &session); */
  Status load_graph_status = LoadGraph(graph_file, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, input_tensor}},
                                   {output_layer_1, output_layer_2}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  /* std::cout<< "reg: " << outputs[0].tensor<float, 4>() << std::endl; */
  std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl;
  return 0;
}
