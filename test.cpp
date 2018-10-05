# include "mtcnn.cpp"

int main(){
  /* string img_file = "./data/22.jpg"; */
  string input_file = "./data/nba.jpg";
  string output_file = "output.jpg";
  float prob_thrd[] = {0.6, 0.6, 0.7};
  float merge_thrd[] = {0.5, 0.7, 0.7, 0.7};
  MTCNN mtcnn;
  mtcnn.Setup(prob_thrd, merge_thrd, 40, 0.709);
  mtcnn.Detect(input_file, output_file);
}
