# src/cpu/Makefile

SUBDIRS := steps

.PHONY: all test clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

test:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir test; \
	done

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
