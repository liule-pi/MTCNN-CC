#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "tensorflow/core/public/session.h"

using tensorflow::Tensor;
using tensorflow::Status;

class Network {
public:
    Network(const std::string& graph_file_path, const std::string& input_node_name, int num,
            const std::string* output_list);
    Status LoadGraph();
    void Forward(tensorflow::Tensor& input_tensor, std::vector<tensorflow::Tensor>* outputs);
private:
    std::unique_ptr<tensorflow::Session> session;
    std::string graph_file;
    std::string input_node;
    int output_number;
    std::string output_node_list[3];
};

Network::Network(const std::string& graph_file_path, const std::string& input_node_name, int num,
                 const std::string* output_list) {
    graph_file = graph_file_path;
    input_node = input_node_name;
    for (int i = 0; i < num; ++i)
    {
        output_node_list[i] = output_list[i];
    }

    Status load_graph_status = LoadGraph();
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        throw std::runtime_error("Load model failed.");
    }
}

Status Network::LoadGraph() {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound(
                "Failed to load compute graph at '", graph_file, "'");
    }
    session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session).Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

void Network::Forward(tensorflow::Tensor& input_tensor, std::vector<tensorflow::Tensor>* outputs){
    auto output_nodes  = {output_node_list[0], output_node_list[1]};
    Status run_status = session->Run({{input_node, input_tensor}},
                output_nodes, {}, outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        throw std::runtime_error("Model forward failed.");
    }
}

int main()
{
  std::string root_dir = "";
  std::string img_file = "./data/test.jpg";
  std::string graph_file = "./data/ckpt/pnet/pnet_frozen.pb";
  std::string input_layer = "pnet/input";
  std::string output_node_list[] = {"pnet/conv4-2/BiasAdd:0", "pnet/prob1:0"};
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

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Network p_net(graph_file, input_layer, 2, output_node_list);
  p_net.Forward(input_tensor, &outputs);

  std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl;
  return 0;
}
