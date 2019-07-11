#general compiling optins
OS=$(shell uname)

OPENCV=opencv

LIB_DUBINS=libDubins.a

CXX=g++

INC= -I./lib/include 
LDFLAGS=-Wl,-rpath,/Users/enrico/opencv/lib/

LIBS=-L./lib -lDubins $(INC)

INCLUDE=include

ifneq (,$(findstring Darwin, $(OS)))
	OPENCV=opencv3
	CXXFLAGS=$(LDFLAGS) -framework OpenCL `pkg-config --cflags tesseract $(OPENCV)` -std=c++11 -Wno-everything -O3
  AR=libtool -static -o
else 
	CXXFLAGS=`pkg-config --cflags tesseract $(OPENCV)` -std=c++11 -Wall -O3
	AR=ar rcs
endif

LDLIBS=`pkg-config --libs tesseract $(OPENCV)` $(LIBS)
MORE_FLAGS=

#general documentation optins
DOXYGEN=doxygen
DOX_CONF_FILE=Doxyfile

MKDIR=mkdir -p

#files that contain code
#dubins and maths are only libraries
SRC=src/calibration.cc\
	src/detection.cc\
	src/unwrapping.cc\
	src/utils.cc

#test files
TEST_SRC=test/maths_test.cc

#object files
OBJ=$(SRC:.cc=.o)

TEST_EXEC=$(TEST_SRC:.cc=.out)

clr=clear && clear && clear

PROJ_HOME = $(shell pwd)

#general function
src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LDLIBS)

test/%.out: test/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@ $< $(LDLIBS)

all: lib bin/ xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/main.out src/main.cc $(LDLIBS)

test: lib bin_test/ $(TEST_EXEC)

lib: lib/$(LIB_DUBINS)

include_local: 
	@rm -rf lib/include
	$(MKDIR) lib
	$(MKDIR) lib/include
	@cp -f src/$(INCLUDE)/*.hh lib/include

lib/libDubins.a: include_local $(OBJ)
	@$(MKDIR) lib
	$(AR) lib/libDubins.a $(OBJ) 

bin/:
	$(MKDIR) bin

bin_test/: bin
	$(MKDIR) bin/test

#compile executables
calibration: bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/$@_run.cc $(LDLIBS)

unwrapping: bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/$@_run.cc $(LDLIBS)

detection: bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/$@_run.cc $(LDLIBS)

#run executables
run:
	./bin/main.out

run_calibration:
	./bin/calibration_run.out

run_unwrapping:
	./bin/unwrapping_run.out

run_detection:
	./bin/detection_run.out

xml generateXML:
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -Wall -O3 -o bin/create_xml.out src/create_xml.cc $(LDLIBS)
	./bin/create_xml.out

#clean lib
clean_lib lib_clean:
	rm -rf lib

#clean objects
clean_obj obj_clean:
	rm -f src/*.o

#clean executables
clean_exec exec_clean:
	rm -rf bin

clean_test test_clean:
	rm -f test/*.out


#clean executables and objects
clean:
	make clean_obj
	make clean_exec
	make clean_test
	make clean_lib

#clean documentation
doc_clean clean_doc:
	rm -rf docs

#make documentation for html and latex (also compile latex)
doc:
	$(MKDIR) docs 
	$(DOXYGEN) $(DOX_CONF_FILE)
ifneq (,$(shell which pdflatex))
	@cd docs/latex && make
endif

