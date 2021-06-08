#add_subdirectory(support)
#add_subdirectory(sutil)

file(GLOB_RECURSE HEADER_FILES *.h*)
file(GLOB_RECURSE SOURCE_FILES *.c*)

add_library(sdk-pt SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(sdk-pt
        PUBLIC
        luminous-ext
        luminous-util
        luminous-gpu
        luminous-core)