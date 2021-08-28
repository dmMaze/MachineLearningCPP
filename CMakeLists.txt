cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(SVM VERSION 0.1)


string(TOLOWER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)

if (COMPILER_ID STREQUAL "msvc")
    set(IS_CLANG_BUILD true)
    set(OPENBLAS_INSTALL_DIR "E:/packages/OpenBLAS/msvc_win32")
    set(EIGEN_PATH "E:/packages/Eigen3/msvc/include/eigen3")
else()
    set(IS_CLANG_BUILD false)
    set(OPENBLAS_INSTALL_DIR "D:/Program Files (x86)/OpenBLAS/")
    set(EIGEN_PATH "D:/Program Files (x86)/Eigen3/include/eigen3")
endif()

file(GLOB SOURCES *.h *.cpp )
include_directories(${EIGEN_PATH})
# include_directories("${OPENBLAS_INSTALL_DIR}/include/openblas")
link_directories("${OPENBLAS_INSTALL_DIR}/lib")

add_executable(main ${SOURCES})

if(IS_CLANG_BUILD)
    target_link_libraries(main openblas)
else()
    target_link_libraries(main libopenblas.a)
endif()
