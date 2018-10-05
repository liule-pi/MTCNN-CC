#ifndef __BBOX_HPP__
#define __BBOX_HPP__

#include <vector>
#include <string>
#include <algorithm>

typedef struct FaceBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceBox;

typedef struct PadBBox {
    bool need_pad;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
} PadBBox;

typedef struct FaceLandmarkPoints {
    float x[5], y[5];
} FaceLandmarkPoints;

typedef struct FaceInfo {
    FaceBox rect;
    float regression[4];
    PadBBox pad;
    FaceLandmarkPoints face_landmark_points;
} FaceInfo;

class BoundingBOX {
public:
    void NonMaximumSuppression(std::vector<FaceInfo>& bounding_boxes, float thresh, char method);
    void BBoxRegress(int stage);
    void BBox2Square();
    void BBoxPadding(int w, int h);
public:
    std::vector<FaceInfo> candidate_bboxes;
    std::vector<FaceInfo> total_bboxes;
private:
    static bool CompareBBox(const FaceInfo& a, const FaceInfo& b);
};
#endif
