#include <iostream>
#include <vector>
#include <cmath>

#define MIN_IMAGE_SIZE 12

void get_scales(std::vector<float>* scales, int h, int w,
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
        min_hw_scaled = std::floor(min_hw * scale);
    }
}

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "useage ./main H W " << std::endl;
    return -1;
  }
  const int MIN_FACE_SIZE = 20;
  double factor = 0.709;

  int image_h = std::stoi(argv[1]);
  int image_w = std::stoi(argv[2]);

  std::vector<float> scales;
  get_scales(&scales, image_h, image_w, MIN_FACE_SIZE, factor);


    for (std::vector<float>::iterator iter = scales.begin();
                                    iter != scales.end(); ++iter) {
        std::cout << "image: " << int(*iter * image_h) << "X"
                  << int(*iter * image_w) << std::endl;
  }
  return 0;
}
