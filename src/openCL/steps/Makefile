# src/openCL/steps/Makefile

SUBDIRS := test

SRCS := *.cpp
OBJS := $(addprefix .lib/,$(SRCS:.cpp=.o))
DEPS := $(addprefix .depends/,$(SRCS:.cpp=.d))

INCLUDES := -I../../../include
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 $(INCLUDES) -MMD -MP

.PHONY: all test clean $(SUBDIRS)

all: $(OBJS) $(SUBDIRS)

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
	rm -rf .lib .depends
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
