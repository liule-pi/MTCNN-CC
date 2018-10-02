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

#define MIN_IMAGE_SIZE 12

typedef struct FaceBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceBox;

typedef struct FaceLandmarkPoints {
    float x[5], y[5];
} FaceLandmarkPoints;

typedef struct FaceInfo {
    FaceBox face_box;
    float regression[4];
    FaceLandmarkPoints face_landmark_points;
} FaceInfo;

void get_scales(std::vector<float>* scales, int h, int w, int mini_face_size, float factor);
