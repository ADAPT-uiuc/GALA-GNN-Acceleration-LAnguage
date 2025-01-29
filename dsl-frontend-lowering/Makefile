# Compiler and tools
CXX = g++
FLEX = flex
BISON = bison

# Compiler flags
CXXFLAGS = -ggdb3

# Source files
LEX_SRC = frontend.l
BISON_SRC = frontend.y

# Output directories
BUILD_DIR = build
IR_DIR = $(BUILD_DIR)/ir

# Generated files
LEX_GEN = $(BUILD_DIR)/lex.yy.c
BISON_GEN_C = $(BUILD_DIR)/frontend.tab.c
BISON_GEN_H = $(BUILD_DIR)/frontend.tab.h

# Output executable
TARGET = frontend

# Default target
all: $(TARGET)

# Rule to build the final executable
$(TARGET): $(BISON_GEN_C) $(LEX_GEN)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(BISON_GEN_C) $(LEX_GEN)

# Rule to generate the Bison parser
$(BISON_GEN_C) $(BISON_GEN_H): $(BISON_SRC) | $(BUILD_DIR)
	$(BISON) -t -v -d -o $(BISON_GEN_C) $(BISON_SRC)

# Rule to generate the Flex scanner
$(LEX_GEN): $(LEX_SRC) | $(BUILD_DIR)
	$(FLEX) -o $(LEX_GEN) $(LEX_SRC)
#   $(FLEX) -o $(LEX_GEN) -d $(LEX_SRC) <-- when debugging the syntax

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up generated files but preserve the IR directory
clean:
	find $(BUILD_DIR) -mindepth 1 -not -path "$(BUILD_DIR)/ir" -not -path "$(BUILD_DIR)/ir/*" -delete
	rm -f $(TARGET)

.PHONY: all clean
