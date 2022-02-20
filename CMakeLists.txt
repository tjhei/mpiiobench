cmake_minimum_required(VERSION 3.0)
Project(mpibigtest)

SET(TARGET "mpibigtest")

find_package(MPI REQUIRED)

message(STATUS "Found MPI version ${MPI_CXX_VERSION_MAJOR}.${MPI_CXX_VERSION_MINOR}")


add_executable(${TARGET} mpi.cc)
target_include_directories(${TARGET} PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(${TARGET} PRIVATE ${MPI_CXX_COMPILE_FLAGS} "-Wall" "-O2")
target_link_libraries(${TARGET} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} "-Wall" "-O2")

IF(MPI_CXX_VERSION_MAJOR GREATER_EQUAL 3)
  add_compile_definitions(${TARGET} WITH_MPI3)
ENDIF()
