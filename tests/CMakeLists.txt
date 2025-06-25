
include_directories(${CMAKE_CURRENT_BINARY_DIR} ../src/frontend)

BISON_TARGET(Parser ../src/frontend/frontend.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/parser.hpp)
FLEX_TARGET(Lexer ../src/frontend/frontend.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(Lexer Parser)

add_executable(gala_inference gala_inference.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
add_executable(gala_train gala_train.cpp ${BISON_Parser_OUTPUTS} ${FLEX_Lexer_OUTPUTS})
