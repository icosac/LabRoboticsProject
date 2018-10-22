#the number is given according to the lectures
#files that contain code
T = main
T2 = sort4Points

#supporting files
S = aaa

#general optins
CXX=g++
CXXFLAGS=`pkg-config --cflags opencv`
LDLIBS=`pkg-config --libs opencv`

#general function
all: run

b build: 
	$(CXX) $(CXXFLAGS) $(T).cpp -o $(T).out $(LDLIBS)
	
#$(CXX) $(CXXFLAGS) -c $(T).cpp
#$(CXX) -o $(T).out $(T).o $(LDLIBS)
r run: build
	./$(T).out

clean: 
	rm -rf *.out *.o
