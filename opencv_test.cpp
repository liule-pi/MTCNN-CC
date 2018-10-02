/* #include <opencv2/core/core.hpp> */
/* #include <opencv2/highgui/highgui.hpp> */
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
  double factor = 0.3;
   Mat img = imread("./data/test.jpg");
   Mat resized;
   resize(img, resized, Size(), factor, factor, 3);
   /* int img_w = img.cols; */
   /* int img_h = img.rows; */
   namedWindow("frame", WINDOW_NORMAL);
   imshow("frame", img);
   waitKey(0);
   imshow("frame", resized);
   waitKey(0);
   imwrite("resized.jpg", resized);
}
