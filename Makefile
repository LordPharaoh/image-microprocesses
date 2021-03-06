all:
	g++ -ggdb `pkg-config --cflags --libs opencv` -std=c++11 src/buoyNew.cpp src/buoyNew.hpp -o buoyNew
	g++ -ggdb `pkg-config --cflags --libs opencv` -lconfig++ -std=c++11 src/color_crop.hpp src/color_crop.cpp src/dropperNew.cpp src/dropperNew.hpp -o dropperNew
clean:
	rm buoyNew
	rm dropperNew	
dropper:
	g++ -ggdb `pkg-config --cflags --libs opencv` -lconfig++ -std=c++11 src/color_crop.hpp src/color_crop.cpp src/dropperNew.cpp src/dropperNew.hpp -o dropperNew
buoy:
	g++ -ggdb `pkg-config --cflags --libs opencv` -std=c++11 src/buoyNew.cpp src/buoyNew.hpp -o buoyNew
	 
