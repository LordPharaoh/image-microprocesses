all:
	g++ -ggdb `pkg-config --cflags --libs opencv` -lconfig++ -std=c++11 src/dropperNew.cpp src/dropperNew.hpp -o dropperNew
	g++ -ggdb `pkg-config --cflags --libs opencv` -std=c++11 src/buoyNew.cpp src/buoyNew.hpp -o buoyNew
clean:
	rm buoyNew
	rm dropperNew	
dropper:
	g++ -ggdb `pkg-config --cflags --libs opencv` -lconfig++ -std=c++11 src/dropperNew.cpp src/dropperNew.hpp -o dropperNew
buoy:
	g++ -ggdb `pkg-config --cflags --libs opencv` -std=c++11 src/buoyNew.cpp src/buoyNew.hpp -o buoyNew
	 
