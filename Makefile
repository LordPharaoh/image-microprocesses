all:
	g++ -ggdb `pkg-config --cflags --libs opencv` -std=c++11 buoyNew.cpp buoyNew.hpp -o buoyNew
clean:
	rm buoyNew
	 
