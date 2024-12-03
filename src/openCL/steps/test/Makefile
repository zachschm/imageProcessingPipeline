# src/openCL/steps/test/Makefile

# Automatically include all .cpp files in the current directory
SRCS := $(wildcard *.cpp)
OBJS := $(addprefix .build/,$(SRCS:.cpp=.o))
DEPS := $(addprefix .build/,$(SRCS:.cpp=.d))

# Include paths relative to the test directory
INCLUDES := -I../../../../include
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 $(INCLUDES) -MMD -MP

# .build directory will hold .o files, .d files, and the executable
BUILD_DIR := .build
TARGET := $(BUILD_DIR)/GrayscaleTest  # Adjust executable name as needed

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) -o $@ $^

# Pattern rule for compiling .cpp files to .o files in the .build directory
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Include dependency files
-include $(DEPS)

test: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
