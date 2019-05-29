#general compiling optins
OS=$(shell uname)
OPENCV=opencv
CXX=g++
LDFLAGS=-Wl,-rpath,/Users/enrico/opencv/lib/

ifneq (,$(findstring Darwin, $(OS)))
	OPENCV=opencv3
	CXXFLAGS=$(LDFLAGS) `pkg-config --cflags tesseract $(OPENCV)` -std=c++11 -Wno-everything -O3
else 
	CXXFLAGS=`pkg-config --cflags tesseract $(OPENCV)` -std=c++11 -Wall -O3
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
	src/utils.cc

#test files
TEST_SRC= test/compare_test.cc\
					test/DubinsFunc.cc\
					test/LSL_test.cc\
					test/maths_test.cc\
					test/scale_to_standard_test.cc\
					test/split_test.cc

#object files
OBJ=$(SRC:.cc=.o)

TEST_EXEC=$(TEST_SRC:.cc=.out)

clr=clear && clear && clear

#general function
src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -c -o $@ $< $(LDLIBS)

test/%.out: test/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@ $< $(LDLIBS)

all: $(OBJ) bin/ xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/main.out $(OBJ) src/main.cc $(LDLIBS)

test: bin_test/ $(TEST_EXEC)

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

