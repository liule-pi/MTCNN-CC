#include "bbox.hpp"

// method, u for IOU, m for IOM;
void BoundingBOX::NonMaximumSuppression(std::vector<FaceInfo>& bounding_boxes, float thresh, char method) {
    std::vector<FaceInfo> temp_bboxes(bounding_boxes);
    bounding_boxes.clear();
    std::sort(temp_bboxes.begin(), temp_bboxes.end(), CompareBBox);
    std::vector<int>::size_type num_bboxes = temp_bboxes.size();
    std::vector<int>::size_type select_index = 0;
    std::vector<int> merge_mask(num_bboxes, 0);

    while(select_index < num_bboxes) {
        while (merge_mask[select_index] == 1)
            if (++select_index == num_bboxes)
                break;
        if (select_index == num_bboxes)
            continue;
        FaceInfo& bbox_s = temp_bboxes[select_index];
        bounding_boxes.push_back(bbox_s);
        float xs1 = bbox_s.rect.x1;
        float xs2 = bbox_s.rect.x2;
        float ys1 = bbox_s.rect.y1;
        float ys2 = bbox_s.rect.y2;
        float area_s = (xs2 - xs1 + 1) * (ys2 - ys1 + 1);
        merge_mask[select_index++] = 1;

        for (std::vector<int>::size_type i = select_index; i < num_bboxes; ++i) {
            if (merge_mask[i] == 1)
                continue;
            FaceInfo& bbox_t = temp_bboxes[i];
            float xt1 = bbox_t.rect.x1;
            float xt2 = bbox_t.rect.x2;
            float yt1 = bbox_t.rect.y1;
            float yt2 = bbox_t.rect.y2;

            float x1 = std::max(xs1, xt1);
            float y1 = std::max(ys1, yt1);
            float x2 = std::min(xs2, xt2);
            float y2 = std::min(ys2, yt2);
            float w = x2 - x1, h = y2 - y1;
            // if not insection
            if ( w <= 0 || h <= 0)
                continue;
            float area_t = (xt2 - xt1 + 1) * (yt2 - yt1 + 1);
            float area_i = w * h;

            switch (method) {
            case 'u':
                if ((area_i) / (area_s + area_t - area_i) > thresh)
                    merge_mask[i] = 1;
            break;
            case 'm':
                if ((area_i) / std::min(area_s, area_t) > thresh)
                    merge_mask[i] = 1;
                break;
            default:
                break;
            }
        }
    }
}

void BoundingBOX::BBoxRegress(int stage) {
    for (std::vector<FaceInfo>::iterator iter = total_bboxes.begin();
                                         iter != total_bboxes.end(); ++iter) {
        float w = iter->rect.x2 - iter->rect.x1;
        float h = iter->rect.y2 - iter->rect.y1;
        w += (stage == 1) ? 0 : 1;
        h += (stage == 1) ? 0 : 1;
        iter->rect.x1 += w * iter->regression[0];
        iter->rect.y1 += h * iter->regression[1];
        iter->rect.x2 += w * iter->regression[2];
        iter->rect.y2 += h * iter->regression[3];
    }
}

void BoundingBOX::BBox2Square() {
    for (std::vector<FaceInfo>::iterator iter = total_bboxes.begin();
                                         iter != total_bboxes.end(); ++iter) {
        float w = iter->rect.x2 - iter->rect.x1;
        float h = iter->rect.y2 - iter->rect.y1;
        float a = w > h ? w : h;
        iter->rect.x1 += (w - a) * 0.5;
        iter->rect.y1 += (h - a) * 0.5;
        iter->rect.x2 += (a - w) * 0.5;
        iter->rect.y2 += (a - h) * 0.5;
    }
}

void BoundingBOX::BBoxPadding(int w, int h){
    for (std::vector<FaceInfo>::iterator iter = total_bboxes.begin();
                                         iter != total_bboxes.end(); ++iter) {
        iter->pad.need_pad = false;
        if (iter->rect.x1 < 1) {
            iter->pad.need_pad = true;
            iter->pad.pad_left = int(1 - iter->rect.x1);
            iter->rect.x1 = 1;
        }
        else {
            iter->pad.pad_left = 0;
            iter->rect.x1 = int(iter->rect.x1);
        }
        if (iter->rect.y1 < 1) {
            iter->pad.need_pad = true;
            iter->pad.pad_top = int(1 - iter->rect.y1);
            iter->rect.y1 = 1;
        }
        else {
            iter->pad.pad_top = 0;
            iter->rect.y1 = int(iter->rect.y1);
        }
        if (iter->rect.x2 > w) {
            iter->pad.need_pad = true;
            iter->pad.pad_right = int(iter->rect.x2 - w);
            iter->rect.x2 = w;
        }
        else {
            iter->pad.pad_right = 0;
            iter->rect.x2 = int(iter->rect.x2);
        }
        if (iter->rect.y2 > h) {
            iter->pad.need_pad = true;
            iter->pad.pad_bottom = int(iter->rect.y2 - h);
            iter->rect.y2 = h;
        }
        else {
            iter->pad.pad_bottom = 0;
            iter->rect.y2 = int(iter->rect.y2);
        }
    }
}

bool BoundingBOX::CompareBBox(const FaceInfo& a, const FaceInfo& b) {
    return a.rect.score > b.rect.score;
}
