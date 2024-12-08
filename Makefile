# Top-Level Makefile

SUBDIRS := src

# Define OpenCV flags
PKG_CONFIG := pkg-config
OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv4)
OPENCV_LDFLAGS := $(shell $(PKG_CONFIG) --libs opencv4)

# Export the OpenCV flags to sub-Makefiles
export OPENCV_CFLAGS
export OPENCV_LDFLAGS

.PHONY: all test clean $(SUBDIRS)

all: main

main:
	$(MAKE) -C src

test: main
	$(MAKE) -C src test

clean:
	$(MAKE) -C src clean
