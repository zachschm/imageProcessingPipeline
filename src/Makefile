# src/Makefile

SUBDIRS := cpu openCL

SRCS := main.cpp BatchManager.cpp Image.cpp Pipeline.cpp
OBJS := $(addprefix .lib/,$(SRCS:.cpp=.o))
DEPS := $(addprefix .depends/,$(SRCS:.cpp=.d))

SUBDIR_OBJS := $(shell find $(SUBDIRS) -name '*.o' -path '*/.lib/*.o')

INCLUDES := -I../include
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 $(INCLUDES) -MMD -MP
LDFLAGS :=

BIN_DIR := ../bin
TARGET := $(BIN_DIR)/ImageProcessingPipeline

.PHONY: all test clean $(SUBDIRS)

all: $(SUBDIRS) $(TARGET)

$(TARGET): $(OBJS) $(SUBDIR_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $^

.lib/%.o: %.cpp
	@mkdir -p .lib .depends
	$(CXX) $(CXXFLAGS) -c $< -o $@
	mv $(patsubst %.o,%.d,$@) .depends/

-include $(DEPS)

$(SUBDIRS):
	$(MAKE) -C $@

test:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir test; \
	done

clean:
	rm -rf $(TARGET) .lib .depends
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

