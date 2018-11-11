#general compiling optins
OS=$(shell uname)

CXX=g++
ifneq (,$(findstring Darwin, $(OS)))
	CXXFLAGS=`pkg-config --cflags tesseract opencv` -std=c++11 -Wno-everything -O3
else 
	CXXFLAGS=`pkg-config --cflags tesseract opencv` -std=c++11 -Wall -O3
endif
LDLIBS=`pkg-config --libs tesseract opencv`
MORE_FLAGS=

#general documentation optins
DOXYGEN=doxygen
DOX_CONF_FILE=Doxyfile

MKDIR=mkdir -p

#files that contain code
SRC=src/calibration.cc\
	src/detection.cc\
	src/unwrapping.cc\
	src/utils.cc

#object files
OBJ=$(SRC:.cc=.o)

#general function
src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) $(DETECTION_OPTIONS) -c -o $@ $< $(LDLIBS)

all: $(OBJ) bin/ xml
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/main.out $(OBJ) src/main.cc $(LDLIBS)

bin/:
	$(MKDIR) bin

#compile executables
calibration: src/calibration.o bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/utils.o src/$@.o src/$@_run.cc $(LDLIBS)

unwrapping: src/unwrapping.o bin/
	$(CXX) $(CXXFLAGS) $(MORE_FLAGS) -o bin/$@_run.out src/utils.o src/$@.o src/$@_run.cc $(LDLIBS)

detection: src/detection.o bin/
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

#clean executables and objects
clean:
	make clean_obj
	make clean_exec
	
#clean documentation
doc_clean clean_doc:
	rm -rf doc

#make documentation for html and latex (also compile latex)
doc:
	$(MKDIR) doc 
	$(DOXYGEN) $(DOX_CONF_FILE)
ifneq (,$(shell which pdflatex))
	@cd docs/latex && make
endif

