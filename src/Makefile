# Check if PROJECT_ROOT is set
ifndef PROJECT_ROOT
$(error PROJECT_ROOT environment variable needs to be set)
endif

# Source files
SRCS := main.cpp BatchManager.cpp Image.cpp Pipeline.cpp \
		openCL/OpenCLManager.cpp \
		$(wildcard openCL/steps/*.cpp)
		
# Object and dependency files
# OBJS := $(addprefix .lib/,$(notdir $(SRCS:.cpp=.o)))
# DEPS := $(addprefix .depends/,$(notdir $(SRCS:.cpp=.d)))

OBJS := $(SRCS:%.cpp=.lib/%.o)
DEPS := $(SRCS:%.cpp=.depends/%.d)


# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -I$(PROJECT_ROOT)/include \
             -I$(PROJECT_ROOT)/src/openCL/utils/ \
             -I$(PROJECT_ROOT)/src/openCL/ \
             -I$(PROJECT_ROOT)/src/cpu/steps/ \
             -I$(PROJECT_ROOT)/src/openCL/steps/ \
             -I.


LDFLAGS := -framework OpenCL -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs


# Directories
BIN_DIR := ../bin
TARGET := $(BIN_DIR)/ImageProcessingPipeline

# Default target
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Object file rule
.lib/%.o: %.cpp
	@mkdir -p $(dir $@) $(dir $(subst .lib,.depends,$@))
	$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) -MM $< > $(subst .lib,.depends,$@:.o=.d)

# Include dependency files
-include $(DEPS)

# Clean rule
clean:
	rm -rf $(BIN_DIR) .lib .depends