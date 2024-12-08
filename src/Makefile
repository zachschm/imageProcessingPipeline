
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
CXXFLAGS := -std=c++17 -I../include \
             -I/Users/zachschmitz/SoftwareEngineering/c++/imageProcessingPipeline/src/openCL/utils/ \
             -I/Users/zachschmitz/SoftwareEngineering/c++/imageProcessingPipeline/src/openCL/ \
             -I/Users/zachschmitz/SoftwareEngineering/c++/imageProcessingPipeline/src/cpu/steps/ \
             -I/Users/zachschmitz/SoftwareEngineering/c++/imageProcessingPipeline/src/openCL/steps/ \
             -I. `pkg-config --cflags opencv4`

LDFLAGS := `pkg-config --libs opencv4` -framework OpenCL

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