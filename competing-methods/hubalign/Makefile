CC=g++
CFLAGS=-I.
ODIR=build
_DEPS := $(shell find . -name '*.h')
DEPS  := $(patsubst ./%,%,$(_DEPS))
OBJS := $(patsubst %.h,$(ODIR)/%.o,$(DEPS))


$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: all clean

all: post-build

post-build: main-build
	@echo "Make complete"

main-build: pre-build 
	@${MAKE} --no-print-directory hubalign

pre-build:
	@mkdir -p build


hubalign: ${ODIR}/HubAlign.o $(OBJS)
	$(CC) -o build/$@ $^ $(DEPS) 



clean: 
	rm -rf ./build