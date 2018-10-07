# C++ tensorflow 实现MTCNN模型

   tensorflow C\++ API相对python来说极不完善， 很难单独完成一个神经网络，从各种资料来看，tensorflow涉及C\++都是通过加载固化的模型图和参数，直接应用模型。 相对python，C\++写的测试速度更快，容易部署。

   C\++写tensorflow有两种build方式：1. 把project建在tensorflow源码文件夹内，用Bazel build，Bazel 会解决依赖问题; 2. 先用安装其他第三方库，再Bazel把Tensorflow源码build一个tensorflow C\++的动态库，之后项目编译的时候自己连接需要的动态库。
   方法1其实是build项目的同时build了整个tensorflow，因此build速度慢，生成的二进制文件最少也有100M，此外还限制了项目位置。我们采用第二种方法。

ref: https://stackoverflow.com/questions/33620794/how-to-build-and-use-google-tensorflow-c-api

## 环境配置

* 安装第三方库：[bazel](https://docs.bazel.build/versions/master/install.html)， [protobuf](https://github.com/protocolbuffers/protobuf)，[eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)，[abseil](https://github.com/abseil/abseil-cpp)，[opencv](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)
   
    编译Tensorflow动态库的时候，要注意Bazel会自己下载build所需的第三方库，为保证Tensorflow的动态库正确连接，需要确保系统安装的第三方库和Bazel下载的第三方库版本相同，`tensorflow/workspace.bzl`上有build时依赖三方库的版本号，系统安装的时候选择相应的版本(最主要是protobuf和eigen)。
   
    opencv 4.0 默认安装的动态库在`/usr/local/lib64`。

* clone tensorflow，build tensorflow_cc.so, 复制需要的头文件、动态库到系统目录

## MTCNN模型静态图及权重固化

   tensorflow提供了一个python脚本[freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)可以将python`Saver.save`生成的权重数据以及`tf.train.write_graph`生成的静态图结合在一起，将模型固化成一个二进制文件，可供python或者C\++调用。

* 保存图和checkpoints

   `data/save_model.py`保存MTCNN模型的三个卷积网络的权重和图，模型的python实现及权重数据来源于[davidsandberg/facenet](https://github.com/davidsandberg/facenet/tree/master/src/align), 这里只是增加了一个保存模型数据和图的函数。

* 固化及测试

    `freeze_graph.py`需要提供权重数据，静态图，输入节点名，输出节点名，运行`data/freeze.sh`将三个网络的模型固化到 `data/ckpt`。
    `data/freeze_check.py` 测试固化是否成功，它用python读取刚固化的模型，前向传播，然后比较结果。
    
ref: https://stackoverflow.com/questions/35508866/tensorflow-different-ways-to-export-and-run-graph-in-c/43639305#43639305

## MTCNN模型实现

本实现主要包括三个类，`mtcnn`，`network`，`bbox` 

* mtcnn

    主要是各个阶段的连接，如图片数据的预处理，原始bounding_boxes的生成，输出等。包含3个`network`的实例P-Net, R-Net, O-Net, 以及一个`bbox`的实例处理面部信息。

* network

    固化卷积网络载入，session建立，以及前向传播。

* bbox

    一个`vector`保存提取的bounding_boxes，以及处理bounding_boxes的函数，包括NMS，回归，Rect2Squara, Padding。

ref：

* load frozen model with c\++: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
* Tensor accessing: https://stackoverflow.com/questions/39379747/import-opencv-mat-into-c-tensorflow-without-copying

## Usage

`./test`

argparse:

* --input_image "data/test.jpg", "image to be processed"
* --output_image "output.jpg", "output image to be saved"
* --min_face 40, "minimum face size to detect"
* --confident_threshold "0.7 0.7 0.7", "confident threshold for P-Net, R-Net, O-Net, separated by a space"
* --nms_merge_threshold "0.4 0.6 0.6 0.7", "NMS merge threshold for stage 1 intra, stage 1, stage 2, stage 3"
* --factor 0.709, "factor for scale pyramid image"

## Example

![](./test_detected.jpg)

![](./nba_detected.jpg)

## Credit

* https://github.com/kpzhang93/MTCNN_face_detection_alignment
* https://github.com/davidsandberg/facenet
* https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image
