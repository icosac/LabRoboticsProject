#https://github.com/KhronosGroup/libclcxx.git

#general compiling optins
OS=$(shell uname)

OPENCV=opencv

LIB_DUBINS=libDubins.a

CXX=g++
INC= -I./lib/include 
LIBS= -L./lib -lDubins
LDFLAGS=-Wl,-rpath,/Users/enrico/opencv/lib/

INCLUDE=include

ifneq (,$(findstring Darwin, $(OS)))
	OPENCV=opencv3
	CXXFLAGS=$(LDFLAGS) -framework OpenCL `pkg-config --cflags tesseract $(OPENCV)` -std=c++11 -Wno-everything -O3
  AR= libtool -static -o
else 
	CXXFLAGS=`pkg-config --cflags tesseract $(OPENCV)` -std=c++11 -Wall -O3
	AR= ar rcs
endif

LDLIBS=`pkg-config --libs tesseract $(OPENCV)`
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
	src/utils.cc\

ifeq ($(OpenCL), TRUE)
	INCLUDE=includeCL
	SRC+=src/openCL.cc
	INV+=-I./libclcxx/include/openclc++
endif


#test files
TEST_SRC= test/dubins_CL.cc\
# 					test/compare_test.cc\
# 					test/DubinsFunc.cc\
# 					test/LSL_test.cc\
# 					test/maths_test.cc\
# 					test/scale_to_standard_test.cc\
# 					test/split_test.cc\
# 					test/prova.cc

#object files
OBJ=$(SRC:.cc=.o)

TEST_EXEC=$(TEST_SRC:.cc=.out)

clr=clear && clear && clear

PROJ_HOME = $(shell pwd)

opencl_lib:
	MKDIR tmp 
	MKDIR libclcxx
	cd tmp && git clone https://github.com/KhronosGroup/libclcxx.git \
	&& cd libclcxx && sed -i .backup '/test/d' CMakeLists.txt&& MKDIR build && cd build \
	&& cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$(PROJ_HOME)/libclcxx .. \
	&& make install 
	rm -rf tmp

#general function
src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) $(INC) -c -o $@ $< $(LDLIBS) $(LIBS)

test/%.out: test/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) $(INC) -o bin/$@ $< $(LDLIBS) $(LIBS)

all: lib bin/ xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) $(INC) -o bin/main.out src/main.cc $(LDLIBS) $(LIBS)

test: bin_test/ lib $(TEST_EXEC)

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
calibration: src/calibration.o src/util.o bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/utils.o src/$@.o src/$@_run.cc $(LDLIBS)

unwrapping: src/unwrapping.o src/util.o bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/utils.o src/$@.o src/$@_run.cc $(LDLIBS)

detection: src/detection.o src/util.o bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/utils.o src/$@.o src/$@_run.cc $(LDLIBS)

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

clean_opencl opencl_clean:
	rm -rf tmp
	rm -rf libclcxx

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

