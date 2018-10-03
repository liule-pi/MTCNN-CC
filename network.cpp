#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "tensorflow/core/public/session.h"

using tensorflow::Tensor;
using tensorflow::Status;
using std::string;

class Network {
public:
    Network(const string& graph_file_path, const string& input_node_name,
            const string& output_prob_node_name, const string& output_reg_node_name);

    Network(const string& graph_file_path, const string& input_node_name,
            const string& output_prob_node_name, const string& output_reg_node_name,
            const string& optput_landmark_node_name);
    void Forward(Tensor& input_tensor, std::vector<Tensor>* outputs);
private:
    Status LoadGraph();
private:
    std::unique_ptr<tensorflow::Session> session;
    int output_num;
    string graph_file;
    string input_node;
    std::vector <string> output_nodes;
};

// P-net, Q-net
Network::Network(const string& graph_file_path, const string& input_node_name,
                 const string& output_prob_node_name, const string& output_reg_node_name){
    graph_file = graph_file_path;
    input_node = input_node_name;
    output_nodes.push_back(output_prob_node_name);
    output_nodes.push_back(output_reg_node_name);

    Status load_graph_status = LoadGraph();
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        throw std::runtime_error("Load model failed.");
    }
}

// O-net
Network::Network(const string& graph_file_path, const string& input_node_name,
                 const string& output_prob_node_name, const string& output_reg_node_name,
                 const string& output_landmark_node_name){
    graph_file = graph_file_path;
    input_node = input_node_name;
    output_nodes.push_back(output_prob_node_name);
    output_nodes.push_back(output_reg_node_name);
    output_nodes.push_back(output_landmark_node_name);

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

void Network::Forward(Tensor& input_tensor, std::vector<Tensor>* outputs){
    Status run_status = session->Run({{input_node, input_tensor}}, output_nodes, {}, outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        throw std::runtime_error("Model forward failed.");
    }
}
