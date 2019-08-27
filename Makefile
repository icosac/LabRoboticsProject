#general compiling options
OS=$(shell uname)

OPENCV=opencv
TESS= #If defined remember to include -D TESS in compiling flags

CUDA_PATH=/usr/local/cuda
CUDA_ARCH=sm_60
# CUDA_PATH=/opt/cuda
# CUDA_ARCH=sm_35

LIB_DUBINS=libDubins.a
INCLUDE=include

CXX=g++

INC=-I./lib/include
LDFLAGS=-Wl,-rpath,/Users/enrico/opencv/lib/

LIBS=-L./lib -lDubins $(INC)

#compiling libs&flags
LDLIBS=$(LIBS) `pkg-config --libs $(TESS) $(OPENCV)`
MORE_FLAGS=

#condition for mac and linux
ifneq (,$(findstring Darwin, $(OS)))
	OPENCV=opencv3
	CXXFLAGS=$(LDFLAGS) `pkg-config --cflags $(TESS) $(OPENCV)` -std=c++11 -Wno-everything -O3
  	AR=libtool -static -o
else 
	CXXFLAGS=`pkg-config --cflags $(TESS) $(OPENCV)` -std=c++11 -O3
	AR=ar rcs
endif

#general documentation optins
DOXYGEN=doxygen
DOX_CONF_FILE=Doxyfile

MKDIR=mkdir -p

#files that contain code
#dubins and maths are only libraries
# SRC=$(wildcard src/*.cc)
SRC=src/utils.cc 
#object files
OBJ=$(subst src/,src/obj/,$(patsubst %.cc, %.o, $(SRC)))

SRC_CUDA=$(wildcard src/cuda/*.cu)
OBJ_CUDA=$(subst src/,src/obj/,$(patsubst %.cu, %.o, $(SRC_CUDA)))

#test files
TEST_SRC= test/prova_CUDA.cc\
					# test/dubins_test.cc\
# 				test/maths_test.cc
TEST_EXEC=$(subst test/,bin/test/,$(patsubst %.cc, %.out, $(TEST_SRC)))

#Run files
RUN=$(wildcard src/run/*.cc)
RUN_EXEC=$(subst src/run/,bin/,$(patsubst %cc, %out, $(RUN)))

clr=clear && clear && clear

PROJ_HOME = $(shell pwd)

##CREATE FILES TARGETS
#Create objects file
src/obj/cuda/%.o: src/cuda/%.cu
	nvcc -G -arch=$(CUDA_ARCH) -dc -rdc=true --default-stream per-thread $(CXXFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LDLIBS)
src/obj/%.o: src/%.cc
	$(CXX) -g $(CXXFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LDLIBS)
#Create executables for testing
bin/test/%.out: test/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o $@ $< $(LDLIBS)
#Create executables for main files.
bin/%.out: src/run/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o $@ $< $(LDLIBS)

##MAIN TARGETS
#make all
all: lib bin/ xml main

#Create test case files
test: lib bin_test/ $(TEST_EXEC)

cuda: cuda_set lib_cuda lib bin/ run_test

cuda_set: obj/
	@$(eval LIBS+= -D CUDA -lDubinsCuda -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcudadevrt)
	$(MKDIR) src/obj/cuda

##Debugging
ECHO:
	@echo "SRC: " $(SRC)
	@echo "OBJ: " $(OBJ)
	@echo "RUN: " $(RUN)
	@echo "RUN_EXEC: " $(RUN_EXEC)
	@echo "TEST_SRC: " $(TEST_SRC)
	@echo "TEST_EXEC: " $(TEST_EXEC)


##LIBRARY TARGETS
#Create library
lib: lib/$(LIB_DUBINS)

#Move all headers file in a folder (included in CXX options)
include_local: 
	@rm -rf lib/include
	$(MKDIR) lib
	$(MKDIR) lib/include
	cp -f src/$(INCLUDE)/*.hh lib/include
#Static library made of objects file
lib/libDubins.a: include_local obj/ $(OBJ)
	$(AR) lib/libDubins.a $(OBJ) 

lib_cuda: lib/libDubinsCuda.a

lib/libDubinsCuda.a: include_local $(OBJ_CUDA)
	nvcc -arch=$(CUDA_ARCH) -dlink $(OBJ_CUDA) -o src/obj/cuda/gpuCode.o
	nvcc --lib -o lib/libDubinsCuda.a $(OBJ_CUDA) src/obj/cuda/gpuCode.o

##CREATE DIRECTORIES
#Create directory bin
bin/:
	$(MKDIR) bin

#Create directory obj
obj/:
	$(MKDIR) src/obj

#Create folder for tests' executables
bin_test/: bin/
	$(MKDIR) bin/test


##CREATE MAIN EXECUTABLES
#Main executable
main: lib bin/ xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@.out src/run/$@.cc $(LDLIBS)
#calibrarion executable
calibration: xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/run/$@_run.cc $(LDLIBS)
#Unwrapping executable
unwrapping: xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/run/$@_run.cc $(LDLIBS)
#Detection executable
detection: xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/run/$@_run.cc $(LDLIBS)

##RUN EXECUTABLES
#Run main program
run: main
	./bin/main.out
#Run tests
run_test: test
	$(TEST_EXEC)
#Run calibration
run_calibration: calibration
	./bin/calibration_run.out
.PHONY: run_calibration

#Run unwrapping
run_unwrapping: unwrapping
	./bin/unwrapping_run.out
.PHONY: run_unwrapping

#Run detection
run_detection: detection
	./bin/detection_run.out
.PHONY: run_detection run_detection

#Generate xml settings file. Deprecated (TODO remove)
xml generateXML: bin/ lib
# 	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/create_xml.out src/create_xml.cc $(LDLIBS)
# 	./bin/create_xml.out


##CLEAN TARGETS
#clean lib
clean_lib lib_clean:
	rm -rf lib

#clean objects
clean_obj obj_clean:
	rm -rf src/obj

#clean executables
clean_exec exec_clean:
	rm -rf bin

#clean executables, libraries and objects
clean:
	make clean_obj
	make clean_exec
	make clean_lib

##DOCUMENTATION TARGETS
#clean documentation
doc_clean clean_doc:
	rm -rf docs

#make documentation for html and latex (also compile latex)
doc:
	$(MKDIR) docs
	@cp .index.html docs/index.html
	$(DOXYGEN) $(DOX_CONF_FILE)
ifneq (,$(shell which pdflatex))
	@cd docs/latex && make
endif

