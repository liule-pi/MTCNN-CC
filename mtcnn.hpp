#ifndef __MTCNN_HPP__
#define __MTCNN_HPP__

#include "bbox.hpp"
#include "network.hpp"

#define MIN_IMAGE_SIZE 12

using tensorflow::Tensor;
using tensorflow::Status;
using std::string;

const string pnet_graph_file = "./data/ckpt/pnet/pnet_frozen.pb";
const string pnet_input_node = "pnet/input";
const std::vector<string> pnet_output_nodes = {"pnet/prob1:0", "pnet/conv4-2/BiasAdd:0"};

const string rnet_graph_file = "./data/ckpt/rnet/rnet_frozen.pb";
const string rnet_input_node = "rnet/input";
const std::vector<string> rnet_output_nodes = {"rnet/prob1:0", "rnet/conv5-2/conv5-2:0"};

const string onet_graph_file = "./data/ckpt/onet/onet_frozen.pb";
const string onet_input_node = "onet/input";
const std::vector<string> onet_output_nodes = {"onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0"};

class MTCNN {
public:
    MTCNN(int mini_face, const float* prob_thrd, const float* merge_thrd, float fac);
    void Detect(const string& input_file, const string& output_file);
private:
    void GetScales(std::vector<float>* scales, int w, int h);
    void GenerateBBox(const std::vector<Tensor>& outputs, int image_w, int image_h, float scale);
    void DrawFaceInfo(const cv::Mat& img, const string& img_output);
    void BatchDetect(const cv::Mat& input_img, int stage,  Network& net, int img_size);
    void PrintDetectInfo();
    void TransposeBBox();
private:
    BoundingBOX bounding_boxes;
    Network p_net, r_net, o_net;
    float prob_threshold[3];
    float bbox_merge_threshold[4];
    int mini_face_size;
    float factor;
};

void swap(float &a, float &b);
#endif
