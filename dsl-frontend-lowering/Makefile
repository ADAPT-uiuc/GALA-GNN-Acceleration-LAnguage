# Makefile for building a C++ project with Flex and Bison

# Compiler and tools
CXX = g++
FLEX = flex
BISON = bison

# Compiler flags
CXXFLAGS = -ggdb3

# Source files
LEX_SRC = frontend.l
BISON_SRC = frontend.y

# Generated files
LEX_GEN = lex.yy.c
BISON_GEN_C = frontend.tab.c
BISON_GEN_H = frontend.tab.h

# Output executable
TARGET = frontend

# Default target
all: $(TARGET)

# Rule to build the final executable
$(TARGET): $(BISON_GEN_C) $(LEX_GEN)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(BISON_GEN_C) $(LEX_GEN)

# Rule to generate the Bison parser
$(BISON_GEN_C) $(BISON_GEN_H): $(BISON_SRC)
	$(BISON) -d $(BISON_SRC)

# Rule to generate the Flex scanner
$(LEX_GEN): $(LEX_SRC)
	$(FLEX) -d $(LEX_SRC)

# Clean up generated files
clean:
	rm -f $(LEX_GEN) $(BISON_GEN_C) $(BISON_GEN_H) $(TARGET)

.PHONY: all clean
