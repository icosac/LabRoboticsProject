#general compiling optins
CXX=g++
CXXFLAGS=`pkg-config --cflags opencv` -Wall -O3
LDLIBS=`pkg-config --libs opencv`

#general documentation optins
DOXYGEN=doxygen
DOX_CONF_FILE=Doxyfile

MKDIR=mkdir -p

#files that contain code
SRC=src/calibration.cc\
	src/detection.cc\
	src/unwrapping.cc

#object files
OBJ=$(SRC:.cc=.o)

#compile objects
src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDLIBS)

#compile executables
all: $(OBJ)
	$(CXX) $(CXXFLAGS) -o bin/main.out $(OBJ) src/main.cc $(LDLIBS)

cmp_calibration: src/calibration.o
	$(CXX) $(CXXFLAGS) -o bin/$@_run.out src/$@.o src/$@_run.cc $(LDLIBS)

cmp_unwrapping: src/unwrapping.o
	$(CXX) $(CXXFLAGS) -o bin/$@_run.out src/$@.o src/$@_run.cc $(LDLIBS)

cmp_detection: src/detection.o
	$(CXX) $(CXXFLAGS) -o bin/$@_run.out src/$@.o src/$@_run.cc $(LDLIBS)

#run executables
run: all
	./bin/main.out

run_calibration: cmp_calibration
	./bin/$@_run.out

run_unwrapping: cmp_unwrapping
	./bin/$@_run.out

run_detection: cmp_detection
	./bin/$@_run.out

#clean objects
clean_obj obj_clean:
	rm -f src/*.o

#clean executables
clean_exec exec_clean:
	rm -f bin/*

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
	@cd doc/latex && make

