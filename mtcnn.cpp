#include "mtcnn.hpp"

MTCNN::MTCNN(int mini_face, const float* prob_thrd, const float* merge_thrd, float fac):
             p_net(pnet_graph_file, pnet_input_node, pnet_output_nodes),
             r_net(rnet_graph_file, rnet_input_node, rnet_output_nodes),
             o_net(onet_graph_file, onet_input_node, onet_output_nodes) {

    for(int i = 0; i < 3; ++i){
       prob_threshold[i] = *prob_thrd++;
    }
    for(int i = 0; i < 4; ++i){
       bbox_merge_threshold[i] = *merge_thrd++;
    }
    mini_face_size = mini_face;
    factor = fac;
}

void MTCNN::Detect(const string& input_file, const string& output_file) {

    cv::Mat input_img = cv::imread(input_file);
    float alpha=0.0078125;
    float mean=127.5;
    cv::Mat working_img;

    input_img.convertTo(working_img, CV_32FC3);
    working_img=(working_img-mean)*alpha;
    working_img = working_img.t();
    cv::cvtColor(working_img, working_img, cv::COLOR_BGR2RGB);

    // create scale pyramid
    int img_w = working_img.cols;
    int img_h = working_img.rows;
    std::vector<float> scales;
    GetScales(&scales, img_w, img_h);
    // test
    /* scales.clear(); */
    /* scales.push_back(1.0); */

    // stage 1
    int scaled_img_h, scaled_img_w;
    cv::Mat scaled_img;
    for (std::vector<float>::iterator iter = scales.begin();
                                      iter != scales.end(); ++iter) {
        float scale = *iter;
        scaled_img_w = std::ceil(scale * img_w);
        scaled_img_h = std::ceil(scale * img_h);
        cv::resize(working_img, scaled_img, cv::Size(scaled_img_w, scaled_img_h), 0, 0);

        // copy working_img to tensor, add a dimension
        tensorflow::Tensor pnet_input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                        {1, scaled_img_h, scaled_img_w, 3}));
        float *ptr = pnet_input_tensor.flat<float>().data();
        cv::Mat tensor_img(scaled_img_h, scaled_img_w, CV_32FC3, ptr);
        scaled_img.convertTo(tensor_img, CV_32FC3);

        std::vector<Tensor> pnet_outputs;
        p_net.Forward(pnet_input_tensor, &pnet_outputs);
        // test
        /* std::cout<< "prob: "<< outputs[1].tensor<float, 4>() << std::endl; */

        // generate bbox for current scale image then process NMS
        GenerateBBox(pnet_outputs, scaled_img_w, scaled_img_h, scale);
        bounding_boxes.NonMaximumSuppression(bounding_boxes.candidate_bboxes, bbox_merge_threshold[0], 'u');
        // add to total_bboxes
        bounding_boxes.total_bboxes.insert(bounding_boxes.total_bboxes.end(),
                       bounding_boxes.candidate_bboxes.begin(), bounding_boxes.candidate_bboxes.end());
    }
    scaled_img.release();
    bounding_boxes.candidate_bboxes.clear();
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[1], 'u');
    std::cout<< bounding_boxes.total_bboxes.size() <<" bboxes were generated at stage 1"<<std::endl;

    bounding_boxes.BBoxRegress(1);
    bounding_boxes.BBox2Square();
    bounding_boxes.BBoxPadding(img_w, img_h);

    // stage 2
    BatchDetect(working_img, 2, r_net, 24);
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[2], 'u');
    std::cout<< bounding_boxes.total_bboxes.size() <<" bboxes were generated at stage 2"<<std::endl;

    bounding_boxes.BBoxRegress(2);
    bounding_boxes.BBox2Square();
    bounding_boxes.BBoxPadding(img_w, img_h);

    // stage 3
    BatchDetect(working_img, 3, o_net, 48);
    bounding_boxes.BBoxRegress(3);
    bounding_boxes.NonMaximumSuppression(bounding_boxes.total_bboxes, bbox_merge_threshold[3], 'm');
    /* bounding_boxes.BBox2Square(); */
    bounding_boxes.BBoxPadding(img_w, img_h);

    //finally, swap back x and y
    TransposeBBox();
    PrintDetectInfo();
    DrawFaceInfo(input_img, output_file);
    working_img.release();
    input_img.release();
}

void MTCNN::BatchDetect(const cv::Mat& input_img, int stage,  Network& net, int img_size) {
    std::vector<int>::size_type bboxes_num = bounding_boxes.total_bboxes.size();

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
                                    {static_cast<long long>(bboxes_num), img_size, img_size, 3}));
    float *ptr = input_tensor.flat<float>().data();

    // crop img, padding, resize and copy to input tensor
    for (std::vector<int>::size_type i = 0; i < bboxes_num; ++i) {
        FaceInfo& bbox = bounding_boxes.total_bboxes[i];
        cv::Mat crop_img = input_img(cv::Range(bbox.rect.y1 - 1, bbox.rect.y2),
                                     cv::Range(bbox.rect.x1 - 1, bbox.rect.x2));
        if(bbox.pad.need_pad)
            cv::copyMakeBorder(crop_img, crop_img, bbox.pad.pad_top, bbox.pad.pad_bottom,
                    bbox.pad.pad_left, bbox.pad.pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::resize(crop_img, crop_img, cv::Size(img_size, img_size), 0, 0);
        cv::Mat tensor_img(img_size, img_size, CV_32FC3, ptr);
        crop_img.convertTo(tensor_img, CV_32FC3);
        ptr += img_size * img_size * 3;
        crop_img.release();
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
            // regression data should be transpose here
            bbox.regression[0] = reg(i, 1);
            bbox.regression[1] = reg(i, 0);
            bbox.regression[2] = reg(i, 3);
            bbox.regression[3] = reg(i, 2);
            if (stage == 3) {
                // swap here
                float h = bbox.rect.x2 - bbox.rect.x1 + 1;
                float w = bbox.rect.y2 - bbox.rect.y1 + 1;
                tensorflow::TTypes<float, 2>::Tensor landmark =  net_outputs[2].tensor<float, 2>();
                for (int j = 0; j < 5; ++j) {
                    // landmark points should be swap too
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
    bounding_boxes.candidate_bboxes.clear();
    auto prob = outputs[0].tensor<float, 4>();
    auto reg = outputs[1].tensor<float, 4>();
    int stride = 2;
    int feature_map_w = std::ceil((image_w - MIN_IMAGE_SIZE)*1.0 / stride) + 1;
    int feature_map_h = std::ceil((image_h - MIN_IMAGE_SIZE)*1.0 / stride) + 1;
    if (feature_map_w * feature_map_h * 2 != prob.size()){
        throw std::runtime_error("feature map size error.");
    }
    for (int h = 0; h < feature_map_h; h++) {
        for (int w = 0; w < feature_map_w; w++) {
            if (prob(0, h, w, 1) >= prob_threshold[0]) {
                FaceInfo bbox;
                bbox.rect.x1 = (int)((w * stride + 1)/scale);
                bbox.rect.y1 = (int)((h * stride + 1)/scale);
                bbox.rect.x2 = (int)((w * stride + MIN_IMAGE_SIZE)/scale);
                bbox.rect.y2 = (int)((h * stride + MIN_IMAGE_SIZE)/scale);
                bbox.rect.score = prob(0, h, w, 1);
                // regression data should be transpose here
                bbox.regression[0] = reg(0, h, w, 1);
                bbox.regression[1] = reg(0, h, w, 0);
                bbox.regression[2] = reg(0, h, w, 3);
                bbox.regression[3] = reg(0, h, w, 2);
                bounding_boxes.candidate_bboxes.push_back(bbox);
            }
        }
    }
}


void MTCNN::TransposeBBox() {
    for (std::vector<FaceInfo>::iterator iter = bounding_boxes.total_bboxes.begin();
                                      iter != bounding_boxes.total_bboxes.end(); ++iter) {
        swap(iter->rect.x1, iter->rect.y1);
        swap(iter->rect.x2, iter->rect.y2);
        for (int i = 0; i < 5; ++i)
            swap(iter->face_landmark_points.x[i], iter->face_landmark_points.y[i]);
    }
}

void MTCNN::PrintDetectInfo() {
    std::cout<< bounding_boxes.total_bboxes.size() <<" faces were detected."<<std::endl;
    for (std::vector<FaceInfo>::iterator iter = bounding_boxes.total_bboxes.begin();
                                      iter != bounding_boxes.total_bboxes.end(); ++iter) {
        std::cout<<"bbox: {"<<int(iter->rect.x1)<<", "<<int(iter->rect.y1)<<", "
                            <<int(iter->rect.x2)<<", "<<int(iter->rect.y2)<<"}\t";

        std::cout<<"left eye: {"<<int(iter->face_landmark_points.x[0])<<", "
                                <<int(iter->face_landmark_points.y[0])<<"}\t"
                <<"right eye: {"<<int(iter->face_landmark_points.x[1])<<", "
                                <<int(iter->face_landmark_points.y[1])<<"}\t";

        std::cout<<"nose: {"<<int(iter->face_landmark_points.x[2])<<", "
                            <<int(iter->face_landmark_points.y[2])<<"}\t";

        std::cout<<"mouth left: {"<<int(iter->face_landmark_points.x[3])<<", "
                                  <<int(iter->face_landmark_points.y[3])<<"}\t"
                <<"mouth right: {"<<int(iter->face_landmark_points.x[4])<<", "
                                  <<int(iter->face_landmark_points.y[4])<<"}"<<std::endl;
    }
}

void MTCNN::DrawFaceInfo(const cv::Mat& img, const string& output_name) {
    for (std::vector<FaceInfo>::iterator iter = bounding_boxes.total_bboxes.begin();
                                      iter != bounding_boxes.total_bboxes.end(); ++iter) {

        cv::rectangle(img, cv::Point(iter->rect.x1, iter->rect.y1),
                           cv::Point(iter->rect.x2, iter->rect.y2), cv::Scalar(255, 0, 0), 1);
        for (int i = 0; i < 5; ++i) {
            cv::Point mark = cv::Point(iter->face_landmark_points.x[i], iter->face_landmark_points.y[i]);
            cv::circle(img, mark, 3, cv::Scalar(0, 0, 255), 1);
        }
    }
    cv::imwrite(output_name, img);
}

void swap(float &a, float &b) {
    float t = a;
    a = b;
    b = t;
}
