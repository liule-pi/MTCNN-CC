# include "mtcnn.cpp"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/init_main.h"

using tensorflow::Flag;

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

  clock_t t0 = clock();
  MTCNN mtcnn(min_face_size, prob_thrd, merge_thrd, factor);
  clock_t t1 = clock();
  std::cout << "Init Model in " << (t1 - t0)*1.0/1000000 << " second." << std::endl;
  mtcnn.Detect(input_image, output_image);
  clock_t t2 = clock();
  std::cout << "Detect face in " << (t2 - t1)*1.0/1000000 << " second." << std::endl;
}
