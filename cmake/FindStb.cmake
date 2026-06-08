find_path(Stb_INCLUDE_DIR
    NAMES stb_image.h
    HINTS
        ${Stb_DIR}
        ENV STB_DIR
        /usr/include
        /usr/local/include
        /opt/local/include
        C:/include
)

if(Stb_INCLUDE_DIR)
    set(Stb_FOUND TRUE)
    message(STATUS "Found Stb: ${Stb_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Could not find Stb (stb_image.h). Please install stb or set Stb_DIR.")
endif()

add_library(Stb INTERFACE IMPORTED)
target_include_directories(Stb INTERFACE ${Stb_INCLUDE_DIR})
