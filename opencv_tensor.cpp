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

Status norm_tensor(const Tensor& input_tensor, const float input_mean,
                               const float input_std, std::vector<Tensor>* out_tensors)
{
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  // use a placeholder to read input data
  auto image_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_UINT8);

  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {"input", input_tensor},
  };

  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

  /* auto dims_expander = ExpandDims(root, float_caster, 0); */

  auto resized = Div(root.WithOpName("resized"), Sub(root, float_caster, {input_mean}), {input_std});
  Transpose(root.WithOpName("transposed"), resized, {0, 2, 1, 3});

  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({inputs}, {"transposed"}, {}, out_tensors));
  return Status::OK();
}

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
  double input_mean = 127.5;
  double input_std = 127.5;
  std::string root_dir = "";
  std::string img_file = "./data/12.jpg";
  std::string graph_file = "./data/ckpt/pnet/pnet_frozen.pb";
  std::string input_layer = "pnet/input";
  std::string output_layer_1 = "pnet/conv4-2/BiasAdd:0";
  std::string output_layer_2 = "pnet/prob1:0";
  // read img with opencv
  cv::Mat input_img = cv::imread(img_file);
  cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);

  // covert to tensor
  tensorflow::Tensor input_tensor(tensorflow::DT_UINT8, tensorflow::TensorShape(
                                  {1, input_img.rows, input_img.cols,3}));
  uint8_t *ptr = input_tensor.flat<tensorflow::uint8>().data();
  cv::Mat tmp_img(input_img.rows, input_img.cols, CV_8UC3, ptr);
  input_img.convertTo(tmp_img, CV_8UC3);

  // norm tensor and transpose w, h
  std::vector<Tensor> norm_img_tensors;
  Status norm_tensor_status = norm_tensor(input_tensor,  input_mean,
                              input_std, &norm_img_tensors);
  if (!norm_tensor_status.ok()) {
    LOG(ERROR) << norm_tensor_status;
    return -1;
  }
  const Tensor& input_img_tensor = norm_img_tensors[0];

  /* std::cout<< "img: " << input_img_tensor.tensor<float, 4>() << std::endl; */

  // load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  std::string graph_path = tensorflow::io::JoinPath(root_dir, graph_file);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, input_img_tensor}},
                                   {output_layer_1, output_layer_2}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  std::cout<< "reg: " << outputs[0].tensor<float, 4>() << std::endl;
  std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl;
  return 0;
}
