#include "bbox.cpp"
#include "network.cpp"

#define MIN_IMAGE_SIZE 12


using tensorflow::Tensor;
using tensorflow::Status;
using std::string;

const string pnet_graph_file = "./data/ckpt/pnet/pnet_frozen.pb";
const string pnet_input_node = "pnet/input";
const string pnet_output_nodes[] = {"pnet/prob1:0", "pnet/conv4-2/BiasAdd:0"};

const string rnet_graph_file = "./data/ckpt/rnet/rnet_frozen.pb";
const string rnet_input_node = "rnet/input";
const string rnet_output_nodes[] = {"rnet/prob1:0", "rnet/conv5-2/conv5-2:0"};

const string onet_graph_file = "./data/ckpt/onet/onet_frozen.pb";
const string onet_input_node = "onet/input";
const string onet_output_nodes[] = {"onet/prob1:0", "onet/conv6-2/conv6-2:0", "onet/conv6-3/conv6-3:0"};


class MTCNN {
public:
    MTCNN();
    void Setup(const float* prob_thrd, const float* merge_thrd, int mini_face, float fac);
    void Detect(const string& img_file);
private:
    void GetScales(std::vector<float>* scales, int w, int h);
    void GenerateBBox(const std::vector<Tensor>& outputs, int image_w, int image_h, float scale);
    void DrawFaceInfo(cv::Mat img, const string& img_output);
    void Batch(const cv::Mat& input_img, int stage,  Network& net, int img_size);
private:
    BoundingBOX bounding_boxes;
    Network p_net, r_net, o_net;
    float prob_threshold[3];
    float bbox_merge_threshold[4];
    int mini_face_size;
    float factor;
};

MTCNN::MTCNN():p_net(pnet_graph_file, pnet_input_node, pnet_output_nodes[0], pnet_output_nodes[1]),
               r_net(rnet_graph_file, rnet_input_node, rnet_output_nodes[0], rnet_output_nodes[1]),
               o_net(onet_graph_file, onet_input_node, onet_output_nodes[0], onet_output_nodes[1],
                     onet_output_nodes[2]){ };

void MTCNN::Setup(const float* prob_thrd, const float* merge_thrd, int mini_face, float fac) {
    for(int i = 0; i < 3; ++i){
       prob_threshold[i] = *prob_thrd++;
    }
    for(int i = 0; i < 4; ++i){
       bbox_merge_threshold[i] = *merge_thrd++;
    }
    mini_face_size = mini_face;
    factor = fac;
}

void MTCNN::Detect(const string& img_file) {
    cv::Mat input_img = cv::imread(img_file);
    cv::Mat input_img_rgb, sample_img;
    cv::cvtColor(input_img, input_img_rgb, cv::COLOR_BGR2RGB);
    sample_img = input_img_rgb.t();

    // create scale pyramid
    int img_w = sample_img.cols;
    int img_h = sample_img.rows;
    std::vector<float> scales;
    GetScales(&scales, img_w, img_h);
    // test
    /* scales.clear(); */
    /* scales.push_back(1.0); */

    // stage 1
    int scaled_h, scaled_w;
    cv::Mat scaled_img;
    for (std::vector<float>::iterator iter = scales.begin();
                                      iter != scales.end(); ++iter) {
        float scale = *iter;
        scaled_w = std::ceil(scale * img_w);
        scaled_h = std::ceil(scale * img_h);
        cv::resize(sample_img, scaled_img, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_AREA);

        // copy sample_img to tensor, add a dimension, and normalization
        tensorflow::Tensor pnet_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                        {1, scaled_h, scaled_w, 3}));
        float *ptr = pnet_input_tensor.flat<float>().data();
        cv::Mat tensor_img(scaled_h, scaled_w, CV_32FC3, ptr);
        scaled_img.convertTo(tensor_img, CV_32FC3, 1/127.5, -1.0);

        std::vector<Tensor> pnet_outputs;
        p_net.Forward(pnet_input_tensor, &pnet_outputs);
        // test
        /* std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl; */

        // generate bbox for current scale image then process NMS
        GenerateBBox(pnet_outputs, scaled_w, scaled_h, scale);
        std::cout<< bounding_boxes.bboxes.size() <<" bboxes were generated before nms at scale "<< scale <<std::endl;
        bounding_boxes.NonMaximumSuppression(bounding_boxes.bboxes, bbox_merge_threshold[0], 'u');
        std::cout<< bounding_boxes.bboxes.size() <<" bboxes were generated after nms at scale "<< scale <<std::endl;
        // add to total_bboxes
        bounding_boxes.total_bboxes.insert(bounding_boxes.total_bboxes.end(),
                       bounding_boxes.bboxes.begin(), bounding_boxes.bboxes.end());
    }
    bounding_boxes.bboxes.clear();
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[1], 'u');
    std::cout<< bounding_boxes.total_bboxes.size() <<" bboxes were generated at stage 1"<<std::endl;

    bounding_boxes.BBoxRegress(1);
    bounding_boxes.BBox2Square();
    bounding_boxes.BBoxPadding(input_img.cols, input_img.rows);

    DrawFaceInfo(input_img, "stage_1.jpg");
    // stage 2
    Batch(input_img_rgb, 2, r_net, 24);
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[2], 'u');
    std::cout<< bounding_boxes.total_bboxes.size() <<" bboxes were generated at stage 2"<<std::endl;

    bounding_boxes.BBoxRegress(2);
    bounding_boxes.BBox2Square();
    bounding_boxes.BBoxPadding(input_img.cols, input_img.rows);

    DrawFaceInfo(input_img, "stage_2.jpg");

    // stage 3
    Batch(input_img_rgb, 3, o_net, 48);
    bounding_boxes.BBoxRegress(3);
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[3], 'm');
    std::cout<< bounding_boxes.total_bboxes.size() <<" bboxes were generated at stage 3"<<std::endl;

    DrawFaceInfo(input_img, "stage_3.jpg");
    /* bounding_boxes.BBox2Square(); */
    bounding_boxes.BBoxPadding(input_img.cols, input_img.rows);

    DrawFaceInfo(input_img, "stage_4.jpg");
}

void MTCNN::Batch(const cv::Mat& input_img, int stage,  Network& net, int img_size) {
    std::vector<int>::size_type bboxes_num = bounding_boxes.total_bboxes.size();

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                    {static_cast<long long>(bboxes_num), img_size, img_size, 3}));
    float *ptr = input_tensor.flat<float>().data();

    // crop img, padding,resize, transpose, normalize, copy to input tensor
    for (std::vector<int>::size_type i = 0; i < bboxes_num; ++i) {
        FaceInfo& bbox = bounding_boxes.total_bboxes[i];
        cv::Mat crop_img = input_img(cv::Range(bbox.rect.y1 - 1, bbox.rect.y2),
                                     cv::Range(bbox.rect.x1 - 1, bbox.rect.x2));
        if(bbox.pad.need_pad)
            cv::copyMakeBorder(crop_img, crop_img, bbox.pad.pad_left, bbox.pad.pad_right,
                    bbox.pad.pad_top, bbox.pad.pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::resize(crop_img, crop_img, cv::Size(img_size, img_size), 0, 0, cv::INTER_AREA);
        crop_img.t();
        cv::Mat tensor_img(img_size, img_size, CV_32FC3, ptr);
        crop_img.convertTo(tensor_img, CV_32FC3, 1/127.5, -1.0);
        ptr += img_size * img_size * 3;
    }

    std::vector<Tensor> net_outputs;
    net.Forward(input_tensor, &net_outputs);
    tensorflow::TTypes<float, 2>::Tensor prob = net_outputs[0].tensor<float, 2>();
    tensorflow::TTypes<float, 2>::Tensor reg = net_outputs[1].tensor<float, 2>();

    std::vector<FaceInfo> temp_bboxes(bounding_boxes.total_bboxes);
    bounding_boxes.total_bboxes.clear();
    for (std::vector<int>::size_type i = 0; i < bboxes_num; ++i) {
        if (prob(i, 1) > prob_threshold[stage - 1]) {
            FaceInfo bbox = temp_bboxes[i];
            bbox.rect.score = prob(i, 1);
            for (int j = 0; j < 4; ++j) {
                bbox.regression[j] = reg(i, j);
            }
            if (stage == 3) {
                float w = bbox.rect.x2 - bbox.rect.x1 + 1;
                float h = bbox.rect.y2 - bbox.rect.y1 + 1;
                tensorflow::TTypes<float, 2>::Tensor landmark =  net_outputs[2].tensor<float, 2>();
                for (int j = 0; j < 5; ++j) {
                    bbox.face_landmark_points.x[j] = bbox.rect.x1 + w * landmark(i, j + 5) - 1;
                    bbox.face_landmark_points.y[j] = bbox.rect.y1 + h * landmark(i, j) - 1;
                }
            }
            bounding_boxes.total_bboxes.push_back(bbox);
        }
    }
}

void MTCNN::GetScales(std::vector<float>* scales, int w, int h) {
    int min_hw = std::min(h, w);
    float m = static_cast<float> (MIN_IMAGE_SIZE) / mini_face_size;
    /* min_hw *= m; */
    (*scales).clear();
    float scale = m;
    int min_hw_scaled = min_hw * scale;
    while (min_hw_scaled >= MIN_IMAGE_SIZE)
    {
        // test
        /* std::cout<<"image size " <<min_hw_scaled << std::endl; */
        (*scales).push_back(scale);
        scale *= factor;
        min_hw_scaled = min_hw * scale;
    }
}

void MTCNN::GenerateBBox(const std::vector<Tensor>& outputs, int image_w, int image_h, float scale) {
    bounding_boxes.bboxes.clear();
    auto prob = outputs[0].tensor<float, 4>();
    auto reg = outputs[1].tensor<float, 4>();
    int stride = 2;
    int feature_map_w = std::ceil((image_w - MIN_IMAGE_SIZE)*1.0 / stride) + 1;
    int feature_map_h = std::ceil((image_h - MIN_IMAGE_SIZE)*1.0 / stride) + 1;
    if (feature_map_w * feature_map_h * 2 != prob.size()){
        throw std::runtime_error("feature map size error.");
    }
    for (int w = 0; w < feature_map_w; w++) {
        for (int h = 0; h < feature_map_h; h++) {
            if (prob(0, h, w, 1) >= prob_threshold[0]) {
                FaceInfo bbox;
                //transpose back by swap x, y
                bbox.rect.y1 = ((w * stride + 1)/scale);
                bbox.rect.x1 = ((h * stride + 1)/scale);
                bbox.rect.y2 = ((w * stride + MIN_IMAGE_SIZE)/scale);
                bbox.rect.x2 = ((h * stride + MIN_IMAGE_SIZE)/scale);
                bbox.rect.score = prob(0, h, w, 1);
                for (int i = 0; i < 4; ++i) {
                    bbox.regression[i] = reg(0, h, w, i);
                }
                bounding_boxes.bboxes.push_back(bbox);
            }
        }
    }
}

void MTCNN::DrawFaceInfo(cv::Mat img, const string& output_name) {
    cv::Mat tmp = img.clone();
    for (std::vector<FaceInfo>::iterator iter = bounding_boxes.total_bboxes.begin();
                                      iter != bounding_boxes.total_bboxes.end(); ++iter) {
        int x = iter->rect.x1;
        int y = iter->rect.y1;
        int w = iter->rect.x2 - x;
        int h = iter->rect.y2 - y;
        cv::Rect r = cv::Rect(x, y, w, h);
        cv::rectangle(tmp, r, cv::Scalar(255, 0, 0), 1);
        for (int i = 0; i < 5; ++i) {
            cv::Point mark = cv::Point(iter->face_landmark_points.x[i], iter->face_landmark_points.y[i]);
            cv::circle(tmp, mark, 3, cv::Scalar(0, 0, 255), 1);
        }
    }
    cv::imwrite(output_name, tmp);
    /* cv::namedWindow("frame", cv::WINDOW_NORMAL); */
    /* cv::imshow("frame", img); */
    /* cv::waitKey(0); */
}

int main(){
  /* string img_file = "./data/22.jpg"; */
  string img_file = "./data/test.jpg";
  MTCNN mtcnn;
  float prob_thrd[] = {0.6, 0.6, 0.5};
  float merge_thrd[] = {0.5, 0.7, 0.7, 0.7};
  mtcnn.Setup(prob_thrd, merge_thrd, 40, 0.709);
  mtcnn.Detect(img_file);
}
