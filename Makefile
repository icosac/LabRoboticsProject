#general optins
CXX=g++
CXXFLAGS=`pkg-config --cflags opencv` -Wall -O3
LDLIBS=`pkg-config --libs opencv`

MKDIR=mkdir -p

#files that contain code
SRC=src/calibration.cc\
	src/detection.cc\
	src/unwrapping.cc

#object files
OBJ=$(SRC:.cc=.o)

#general function
all: $(OBJ)
	$(CXX) $(CXXFLAGS) -o bin/main.out $(OBJ) src/main.cc $(LDLIBS)

src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $< $(LDLIBS)

calibration: src/calibration.o
	$(CXX) $(CXXFLAGS) -o bin/$@_run.out src/$@.o src/$@_run.cc $(LDLIBS)
	./bin/$@_run.out

unwrapping: src/unwrapping.o
	$(CXX) $(CXXFLAGS) -o bin/$@_run.out src/$@.o src/$@_run.cc $(LDLIBS)
	./bin/$@_run.out

detection: src/detection.o
	$(CXX) $(CXXFLAGS) -o bin/$@_run.out src/$@.o src/$@_run.cc $(LDLIBS)
	./bin/$@_run.out

run:
	./bin/main.out

clean:
	rm -f bin/*
	rm -f src/*.o

xml generateXML:
	$(CXX) $(CXXFLAGS) -Wall -O3 -o bin/create_xml.out src/create_xml.cc $(LDLIBS)
	./bin/create_xml.out