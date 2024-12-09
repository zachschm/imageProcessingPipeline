# Top-Level Makefile

SUBDIRS := src

.PHONY: all test clean $(SUBDIRS)

all: main

main:
	$(MAKE) -C src

test: main
	$(MAKE) -C src test

clean:
	$(MAKE) -C src clean
