cmake_minimum_required(VERSION 3.16)
project(ex4_amir_rosen15)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(ex4_amir_rosen15
        Activation.cpp
        Activation.h
        Dense.cpp
        Dense.h
        #main.cpp
        Matrix.cpp
        Matrix.h
        MlpNetwork.cpp
        MlpNetwork.h
        #oh_tests_const.cpp
        #presubmit.cpp
        #maino.cpp
        #oh_tests_exceptions.cpp
        #main2.cpp
        test.cpp
        test.h
        test_activation.cpp
        test_activation.h
        test_dense.cpp
        test_dense.h
        test_matrix.cpp
        test_matrix.h
        README.md)
