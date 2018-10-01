### Makefile for building tensorflow application
# HDRS: -I, remember to include eigen3 and tf libs
# LDLIBS : -L, path of folder where libtensorflow_cc.so exist
# LIBS: -l, name of lib.so

SRCS_DIR = .
OBJS_DIR = .
EXE = main

CC = g++
CFLAGS =-std=c++11 -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w

# HDRS = -I/usr/local/include/tf
# HDRS += -I/usr/local/include/tf/bazel-genfiles
# HDRS += -I/usr/include/eigen3
# HDRS += -I/usr/local/include/abseil-cpp

LDLIBS  = -L/usr/local/lib
# LIBS = -lprotobuf
# LIBS += -ltensorflow_cc
# LIBS += -ltensorflow_framework


INPUT_FILE = $(SRCS_DIR)/main.cpp
OBJET_FILE = $(OBJS_DIR)/$(EXE)

$(EXE):
	$(CC) $(CFLAGS) $(INPUT_FILE) -o $(OBJET_FILE) $(HDRS) $(LDLIBS) $(LIBS)

clean:
	rm -f $(OBJET_FILE)

