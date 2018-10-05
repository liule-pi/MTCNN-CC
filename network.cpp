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
            const std::vector<string>& output_nodes_names);
    ~Network();
    void Forward(Tensor& input_tensor, std::vector<Tensor>* outputs);
private:
    Status LoadGraph();
private:
    std::unique_ptr<tensorflow::Session> session;
    int output_num;
    string graph_file;
    string input_node;
    std::vector<string> output_nodes;
};

Network::Network(const string& graph_file_path, const string& input_node_name,
                 const std::vector<string>& output_nodes_names){
    graph_file = graph_file_path;
    input_node = input_node_name;
    output_nodes = output_nodes_names;

    Status load_graph_status = LoadGraph();
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        throw std::runtime_error("Load model failed.");
    }
}

Network::~Network() {
    Status session_close_status =  session->Close();
    if (!session_close_status.ok()) {
        LOG(ERROR) << "session close failed: " << session_close_status;
    }
    session.reset();
    output_nodes.clear();
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
    Status session_create_status = session->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
    delete &graph_def;
}

void Network::Forward(Tensor& input_tensor, std::vector<Tensor>* outputs){
    Status run_status = session->Run({{input_node, input_tensor}}, output_nodes, {}, outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        throw std::runtime_error("Model forward failed.");
    }
}
