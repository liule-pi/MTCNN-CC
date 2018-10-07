ROOT = .
LIBDIR = lib

BIN_SRCS = test.cpp
COMM_SRCS += mtcnn.cpp network.cpp bbox.cpp

CXX := g++
CXXFLAGS += -Wall  -ggdb -std=c++11

HDRS += -I/usr/local/include/opencv4
HDRS += -I/usr/local/include/tf -I/usr/local/include/tf/bazel-genfiles
HDRS += -I/usr/include/eigen3
HDRS += -I/usr/local/include/abseil-cpp

CXXFLAGS += $(HDRS)

LIBS +=-Wl,-rpath,$(ROOT)/$(LIBDIR) -L$(ROOT)/$(LIBDIR)
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
LIBS += -ltensorflow_cc -ltensorflow_framework

COMM_OBJS=$(COMM_SRCS:.cpp=.o)
BIN_OBJS=$(BIN_SRCS:.cpp=.o)
BIN_EXES=$(BIN_SRCS:.cpp=)


default : $(BIN_EXES)

$(BIN_EXES) : $(COMM_OBJS)

$(BIN_EXES):%:%.o


%:%.o
	$(CXX) $< -o $@ $(LDFLAGS) $(COMM_OBJS) $(LIBS)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	-@rm -f $(BIN_EXES) *.o

.PHONY : all clean
