#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cerr << "useage ./main H W " << std::endl;
    return -1;
  }
  const int MIN_FACE_SIZE = 20;
  const int MIN_IMAGE_SIZE = 12;
  double factor = 0.709;

  int image_h = std::stoi(argv[1]);
  int image_w = std::stoi(argv[2]);
  int min_hw = std::min(image_h, image_w);
  double m = static_cast<double> (MIN_IMAGE_SIZE) / MIN_FACE_SIZE;
  min_hw *= m;
  std::vector<double> scales;
  size_t scales_count = 0;
  while (min_hw >= MIN_IMAGE_SIZE)
  {
    scales.push_back(m * std::pow(factor, scales_count++));
    min_hw *= factor;
  }
  for(std::vector<int>::size_type i = 0; i < scales_count; ++i)
  {
    std::cout << "image: " << int(scales[i] * image_h) << "X"
              << int(scales[i] * image_w) << std::endl;
  }
  return 0;
}
