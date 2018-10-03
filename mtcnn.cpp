#include "bbox.cpp"
#include "network.cpp"

#define MIN_IMAGE_SIZE 12


using tensorflow::Tensor;
using tensorflow::Status;
using std::string;

class MTCNN {
public:
    MTCNN();
    void SetPara(const float* prob_thrd, const float* merge_thrd, int mini_face, float fac);
    void Detect(const string& img_file);
    void GetScales(std::vector<float>* scales, int w, int h);
    void GenerateBBox(const std::vector<Tensor>& outputs, int image_w, int image_h, float scale);
    void DrawFaceInfo(cv::Mat img);
    BoundingBOX bounding_boxes;
private:
    Network p_net;
    float prob_threshold[3];
    float bbox_merge_threshold[4];
    int mini_face_size;
    float factor;
};

MTCNN::MTCNN():p_net("./data/ckpt/pnet/pnet_frozen.pb", "pnet/input",
                  "pnet/conv4-2/BiasAdd:0", "pnet/prob1:0"){ };

void MTCNN::SetPara(const float* prob_thrd, const float* merge_thrd, int mini_face, float fac) {
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
    cv::Mat sample_img;
    cv::cvtColor(input_img, sample_img, cv::COLOR_BGR2RGB);
    sample_img = sample_img.t();

    // create scale pyramid
    int img_w = sample_img.cols;
    int img_h = sample_img.rows;
    std::vector<float> scales;
    GetScales(&scales, img_w, img_h);
    // test
    /* scales.clear(); */
    /* scales.push_back(1.0); */

    // stage_1
    int scaled_h, scaled_w;
    cv::Mat scaled_img;
    for (std::vector<float>::iterator iter = scales.begin();
                                      iter != scales.end(); ++iter) {
        float scale = *iter;
        scaled_w = std::ceil(scale * img_w);
        scaled_h = std::ceil(scale * img_h);
        cv::resize(sample_img, scaled_img, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_AREA);

        // copy sample_img to tensor, add a dimension, and normalization
        tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                        {1, scaled_h, scaled_w, 3}));
        float *ptr = input_tensor.flat<float>().data();
        cv::Mat tensor_img(scaled_h, scaled_w, CV_32FC3, ptr);
        scaled_img.convertTo(tensor_img, CV_32FC3, 1/127.5, -1.0);

        std::vector<Tensor> outputs;
        p_net.Forward(input_tensor, &outputs);
        // test
        /* std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl; */

        // generate bbox for current scale image then process NMS
        GenerateBBox(outputs, scaled_w, scaled_h, scale);
        std::cout<< bounding_boxes.bboxes.size() <<" bboxes were generated before nms at scale "<< scale <<std::endl;
        bounding_boxes.NonMaximumSuppression(bounding_boxes.bboxes, bbox_merge_threshold[0], 'u');
        std::cout<< bounding_boxes.bboxes.size() <<" bboxes were generated after nms at scale "<< scale <<std::endl;
        // add to total_bboxes
        bounding_boxes.total_bboxes.insert(bounding_boxes.total_bboxes.end(),
                       bounding_boxes.bboxes.begin(), bounding_boxes.bboxes.end());
    }
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[1], 'u');
    std::cout<< bounding_boxes.total_bboxes.size() <<" bboxes were generated at stage 1"<<std::endl;

    bounding_boxes.BBoxRegress(1);
    bounding_boxes.BBox2Square();

    DrawFaceInfo(input_img);

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
    auto reg = outputs[0].tensor<float, 4>();
    auto prob = outputs[1].tensor<float, 4>();
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

void MTCNN::DrawFaceInfo(cv::Mat img) {
    for (std::vector<FaceInfo>::iterator iter = bounding_boxes.total_bboxes.begin();
                                      iter != bounding_boxes.total_bboxes.end(); ++iter) {
        int x = iter->rect.x1;
        int y = iter->rect.y1;
        int w = iter->rect.x2 - x;
        int h = iter->rect.y2 - y;
        cv::Rect r = cv::Rect(x, y, w, h);
        cv::rectangle(img, r, cv::Scalar(255, 0, 0), 1, 8, 0);
    }
    cv::imwrite("output.jpg", img);
    /* cv::namedWindow("frame", cv::WINDOW_NORMAL); */
    /* cv::imshow("frame", img); */
    /* cv::waitKey(0); */
}

int main(){
  /* string img_file = "./data/22.jpg"; */
  string img_file = "./data/test.jpg";
  MTCNN mtcnn;
  float prob_thrd[] = {0.95, 0.7, 0.7};
  float merge_thrd[] = {0.3, 0.7, 0.7, 0.7};
  mtcnn.SetPara(prob_thrd, merge_thrd, 20, 0.709);
  mtcnn.Detect(img_file);
}
