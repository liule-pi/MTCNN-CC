### Makefile for building tensorflow application
# HDRS: -I, remember to include eigen3 and tf libs
# LIBDIR:  path of folder where libtensorflow_cc.so exist
# LIBS: -l, name of lib.so

SRCS_DIR = .
OBJS_DIR = .
EXE = main

CC = g++
CFLAGS =-std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w


HDRS += -I/usr/local/include/opencv4
HDRS += -I/usr/local/include/tf -I/usr/local/include/tf/bazel-genfiles
HDRS += -I/usr/include/eigen3
HDRS += -I/usr/local/include/abseil-cpp

LIBDIR = /usr/local/lib
LDFLAGS = -L$(LIBDIR)

LDLIBS += -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
# LDLIBS += -lprotobuf
LDLIBS += -ltensorflow_cc -ltensorflow_framework


INPUT_FILE = $(SRCS_DIR)/test.cpp
OBJET_FILE = $(OBJS_DIR)/$(EXE)

$(EXE):
	$(CC) $(CFLAGS) $(INPUT_FILE) -o $(OBJET_FILE) $(HDRS) $(LDFLAGS) $(LDLIBS)

clean:
	-@rm -f $(OBJET_FILE)

run:
	$(OBJET_FILE) ./model/nn_model_frozen.pb
