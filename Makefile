#general compiling options
OS=$(shell uname)

OPENCV=opencv
TESS= #If defined remember to include -D TESS in compiling flags

LIB_DUBINS=libDubins.a
INCLUDE=include

CXX=g++

INC=-I./lib/include 
LDFLAGS=-Wl,-rpath,/Users/enrico/opencv/lib/

LIBS=-L./lib -lDubins $(INC)

#condition for mac and linux
ifneq (,$(findstring Darwin, $(OS)))
	OPENCV=opencv3
	CXXFLAGS=$(LDFLAGS) `pkg-config --cflags $(TESS) $(OPENCV)` -std=c++11 -Wno-everything -O3
  	AR=libtool -static -o
else 
	CXXFLAGS=`pkg-config --cflags $(TESS) $(OPENCV)` -std=c++11 -Wall -O3
	AR=ar rcs
endif

#compiling libs&flags
LDLIBS=$(LIBS) `pkg-config --libs $(TESS) $(OPENCV)` 
MORE_FLAGS=

#general documentation optins
DOXYGEN=doxygen
DOX_CONF_FILE=Doxyfile

MKDIR=mkdir -p

#files that contain code
#dubins and maths are only libraries
SRC=$(wildcard src/*.cc)

#test files
TEST_SRC=\
		test/prova.cc\
# 		test/maths_test.cc\

#object files
# OBJ=$(SRC:.cc=.o)
OBJ=$(subst src/,src/obj/,$(patsubst %.cc, %.o, $(SRC)))

TEST_EXEC=$(TEST_SRC:.cc=.out)

clr=clear && clear && clear

PROJ_HOME = $(shell pwd)

#general functions of the make
src/obj/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LDLIBS)

test/%.out: test/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@ $< $(LDLIBS)

all: ECHO lib bin/ xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/main.out src/main.cc $(LDLIBS)

ECHO: 
	@echo $(OBJ)

test: lib bin_test/ $(TEST_EXEC)

lib: lib/$(LIB_DUBINS)

include_local: 
	@rm -rf lib/include
	$(MKDIR) lib
	$(MKDIR) lib/include
	cp -f src/$(INCLUDE)/*.hh lib/include

lib/libDubins.a: include_local obj/ $(OBJ)
	$(AR) lib/libDubins.a $(OBJ) 

bin/:
	$(MKDIR) bin

obj/:
	$(MKDIR) src/obj

bin_test/: bin
	$(MKDIR) bin/test

#compile executables
calibration: xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/$@_run.cc $(LDLIBS)

unwrapping: xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/$@_run.cc $(LDLIBS)

detection: xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/$@_run.cc $(LDLIBS)

#run executables
run:
	./bin/main.out

run_test:
	./bin/test/prova.out

run_calibration: calibration
	./bin/calibration_run.out
.PHONY: run_calibration

run_unwrapping: unwrapping
	./bin/unwrapping_run.out

run_detection: detection
	./bin/detection_run.out

xml generateXML: bin/ lib
# 	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/create_xml.out src/create_xml.cc $(LDLIBS)
# 	./bin/create_xml.out

#clean lib
clean_lib lib_clean:
	rm -rf lib

#clean objects
clean_obj obj_clean:
	rm -rf src/obj

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

