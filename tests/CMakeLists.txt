
include_directories(${CMAKE_CURRENT_BINARY_DIR} ../src/frontend)

#BISON_TARGET(Parser ../src/frontend/frontend.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.tab.hpp)
#FLEX_TARGET(Lexer ../src/frontend/frontend.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
#BISON_TARGET(Parser ../src/frontend/frontend.y ${CMAKE_CURRENT_BINARY_DIR}/frontend.tab.c DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/frontend.tab.h)
BISON_TARGET(Parser ../src/frontend/frontend.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.hpp)
FLEX_TARGET(Lexer ../src/frontend/frontend.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

add_executable(gcn_ir gala_gcn_IR.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
#add_executable(gin_ir gala_gin_IR.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
#add_executable(gat_ir gala_gat_IR.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})

add_executable(gala gala.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
