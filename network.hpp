#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__

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
#endif
