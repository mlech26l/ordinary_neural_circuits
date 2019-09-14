
SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin

TARGET_LIB   = pybnn.so

OPT = -O3 -march=native
CXX       = g++
# compiling flags here

# Change line 13/14 and 31/32 to switch between python 3.5 and python 2.7
# CFLAGS   =  -pthread -std=c++11 $(OPT) -Wall -I.  -I/usr/include/python3.5m/
CFLAGS   =  -pthread -std=c++11 $(OPT) -Wall -I.  -I/usr/include/python3.6m/
# CFLAGS   =  -pthread -std=c++11 $(OPT) -Wall -I.  -I/usr/include/python2.7/


LINKER   = g++ -o
# linking flags here
LFLAGS   = -pthread -Wall $(OPT) -I.

SOURCES  := $(wildcard $(SRCDIR)/*.cpp)
INCLUDES := $(wildcard $(SRCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

rm       = rm -f


all: $(BINDIR)/$(TARGET_LIB)

$(BINDIR)/$(TARGET_LIB): $(OBJECTS)
	g++ -shared $(LFLAGS) -Wl,-soname,$(TARGET_LIB) -o bin/$(TARGET_LIB) $(OBJECTS) -lpython3.6m -lboost_python-py36
	# g++ -shared $(LFLAGS) -Wl,-soname,$(TARGET_LIB) -o bin/$(TARGET_LIB) $(OBJECTS) -lpython3.5m -lboost_python-py35
	# g++ -shared $(LFLAGS) -Wl,-soname,$(TARGET_LIB) -o bin/$(TARGET_LIB) $(OBJECTS) -lpython2.7 -lboost_python-py27

	@echo "Built lib successfully!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CFLAGS) -c -fPIC $< -o $@
	@echo "Compiled "$<" successfully!"

clean:
	rm -f bin/$(TARGET_LIB)
	rm -f obj/*.o
