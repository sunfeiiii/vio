cmake_minimum_required(VERSION 2.8)

project(vio)

set(CMAKE_CXX_FLAGS "-std=c++17")

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(${realsense2_INCLUDE_DIRS})

include_directories(${EIGEN3_INCLUDE_DIR})
message("eigen" ${G2O_INCLUDE_DIRS} )

include_directories(${G2O_INCLUDE_DIRS} "usr/local/include/g2o")

include_directories(${PCL_INCLUDE_DIRS})

set(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

add_executable(vio src/main.cpp)

target_link_libraries(vio ${OpenCV_LIBS} ${realsense2_LIBRARY} ${G2O_LIBS} ${PCL_LIBRARIES})