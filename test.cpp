#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/init_main.h"
#include <sys/time.h>

#include "mtcnn.hpp"

using tensorflow::Flag;

unsigned long get_cur_time(void)
{
    struct timeval tv;
    unsigned long t_ms;
    gettimeofday(&tv,NULL);
    t_ms = tv.tv_sec*1000+tv.tv_usec/1000;
    return t_ms;
}

int main(int argc, char* argv[]) {
  string input_image = "./data/test.jpg";
  string output_image = "output.jpg";
  int min_face_size = 40;
  string confident_threshold = "0.7 0.7 0.7";
  string nms_merge_threshold = "0.4 0.6 0.6 0.7";
  float factor = 0.709;

  std::vector<Flag> flag_list = {
      Flag("input_image", &input_image, "image to be processed"),
      Flag("output_image", &output_image, "output image to be saved"),
      Flag("min_face", &min_face_size, "minimum face size to detect"),
      Flag("confident_threshold", &confident_threshold, "confident threshold for P-Net, R-Net, O-Net, separated by a space"),
      Flag("nms_merge_threshold", &nms_merge_threshold, "NMS merge threshold for stage 1 intra, stage 1, stage 2, stage 3"),
      Flag("factor", &factor, "factor for scale pyramid image"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  float prob_thrd[3];
  float merge_thrd[4];
  string threshold;

  std::istringstream prob_stream(confident_threshold);
  for (int i = 0; i < 3; ++i) {
      prob_stream >> threshold;
      prob_thrd[i] = stof(threshold);
  }
  std::istringstream merge_stream(nms_merge_threshold);
  for (int i = 0; i < 4; ++i) {
      merge_stream >> threshold;
      merge_thrd[i] = stof(threshold);
  }

  unsigned long t0 = get_cur_time();
  MTCNN mtcnn(min_face_size, prob_thrd, merge_thrd, factor);
  unsigned long t1 = get_cur_time();
  std::cout << "Init Model in " << (t1 - t0) << " millisecond." << std::endl;
  mtcnn.Detect(input_image, output_image);
  unsigned long t2 = get_cur_time();
  std::cout << "Detect face in " << (t2 - t1) << " millisecond." << std::endl;
}
