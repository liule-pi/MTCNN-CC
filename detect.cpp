#include "detect.h"


using tensorflow::Tensor;
using tensorflow::Status;


// image read from file
void detect(const cv::Mat& image, int mini_face_size, float* threshold, double factor) {
    cv::Mat sample_img;
    cv::cvtColor(image, sample_img, cv::COLOR_BGR2RGB);
    sample_img = sample_img.t();

    int img_w = sample_img.cols;
    int img_h = sample_img.rows;
    std::vector<float> scales;
    get_scales(&scales, img_w, img_h, mini_face_size, factor);

    int scaled_h, scaled_w;
    cv::Mat scaled_img;
    for (std::vector<float>::iterator iter = scales.begin();
                                      iter != scales.end(); ++iter) {
        scaled_w = std::ceil(*iter * img_w);
        scaled_h = std::ceil(*iter * img_h);
        cv::resize(sample_img, scaled_img, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_AREA);

        // copy sample_img to tensor, add a dimension, normalization
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                        {1, scaled_h, scaled_w, 3}));
        float *ptr = input_tensor.flat<float>().data();
        cv::Mat tensor_img(scaled_h, scaled_w, CV_32FC3, ptr);
        scaled_img.convertTo(tensor_img, CV_32FC3, 1/127.5, -1.0);
    }
}



void get_scales(std::vector<float>* scales, int w, int h,
                int mini_face_size, float factor){
    int min_hw = std::min(h, w);
    float m = static_cast<float> (MIN_IMAGE_SIZE) / mini_face_size;
    /* min_hw *= m; */
    (*scales).clear();
    float scale = m;
    int min_hw_scaled = min_hw * scale;
    while (min_hw_scaled >= MIN_IMAGE_SIZE)
    {
        (*scales).push_back(scale);
        scale *= factor;
        min_hw_scaled = min_hw * scale;
    }
}


Status LoadGraph(const std::string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}




int main() {
    std::string root_dir = "";
    std::string img_file = "./data/test.jpg";
    std::string graph_file = "./data/ckpt/pnet/pnet_frozen.pb";
    std::string input_layer = "pnet/input";
    std::string output_layer_1 = "pnet/conv4-2/BiasAdd:0";
    std::string output_layer_2 = "pnet/prob1:0";

    // read img with opencv
    cv::Mat input_img = cv::imread(img_file);

    //load P-net
    std::unique_ptr<tensorflow::Session> session;
    std ::string graph_path = tensorflow::io::JoinPath(root_dir, graph_file);
    Status load_graph_status = LoadGraph(graph_path, &session);
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
    std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl;
    return 0;
}
