# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/cmake/bin/cmake

# The command to remove a file.
RM = /usr/local/cmake/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sf/Desktop/vio

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sf/Desktop/vio/build

# Include any dependencies generated for this target.
include CMakeFiles/vio.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/vio.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/vio.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vio.dir/flags.make

CMakeFiles/vio.dir/src/main.cpp.o: CMakeFiles/vio.dir/flags.make
CMakeFiles/vio.dir/src/main.cpp.o: /home/sf/Desktop/vio/src/main.cpp
CMakeFiles/vio.dir/src/main.cpp.o: CMakeFiles/vio.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sf/Desktop/vio/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vio.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/vio.dir/src/main.cpp.o -MF CMakeFiles/vio.dir/src/main.cpp.o.d -o CMakeFiles/vio.dir/src/main.cpp.o -c /home/sf/Desktop/vio/src/main.cpp

CMakeFiles/vio.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vio.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sf/Desktop/vio/src/main.cpp > CMakeFiles/vio.dir/src/main.cpp.i

CMakeFiles/vio.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vio.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sf/Desktop/vio/src/main.cpp -o CMakeFiles/vio.dir/src/main.cpp.s

# Object files for target vio
vio_OBJECTS = \
"CMakeFiles/vio.dir/src/main.cpp.o"

# External object files for target vio
vio_EXTERNAL_OBJECTS =

vio: CMakeFiles/vio.dir/src/main.cpp.o
vio: CMakeFiles/vio.dir/build.make
vio: /usr/local/lib/libopencv_gapi.so.4.6.0
vio: /usr/local/lib/libopencv_highgui.so.4.6.0
vio: /usr/local/lib/libopencv_ml.so.4.6.0
vio: /usr/local/lib/libopencv_objdetect.so.4.6.0
vio: /usr/local/lib/libopencv_photo.so.4.6.0
vio: /usr/local/lib/libopencv_stitching.so.4.6.0
vio: /usr/local/lib/libopencv_video.so.4.6.0
vio: /usr/local/lib/libopencv_videoio.so.4.6.0
vio: /opt/ros/melodic/lib/librealsense2.so.2.50.0
vio: /usr/lib/x86_64-linux-gnu/libboost_system.so
vio: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
vio: /usr/lib/x86_64-linux-gnu/libboost_thread.so
vio: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
vio: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
vio: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
vio: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
vio: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
vio: /usr/lib/x86_64-linux-gnu/libboost_regex.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_common.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
vio: /usr/lib/libOpenNI.so
vio: /usr/lib/libOpenNI2.so
vio: /usr/lib/x86_64-linux-gnu/libfreetype.so
vio: /usr/lib/x86_64-linux-gnu/libz.so
vio: /usr/lib/x86_64-linux-gnu/libexpat.so
vio: /usr/lib/x86_64-linux-gnu/libpython2.7.so
vio: /usr/lib/libvtkWrappingTools-6.3.a
vio: /usr/lib/x86_64-linux-gnu/libjpeg.so
vio: /usr/lib/x86_64-linux-gnu/libpng.so
vio: /usr/lib/x86_64-linux-gnu/libtiff.so
vio: /usr/lib/x86_64-linux-gnu/libproj.so
vio: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
vio: /usr/lib/x86_64-linux-gnu/libsz.so
vio: /usr/lib/x86_64-linux-gnu/libdl.so
vio: /usr/lib/x86_64-linux-gnu/libm.so
vio: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
vio: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
vio: /usr/lib/x86_64-linux-gnu/libnetcdf.so
vio: /usr/lib/x86_64-linux-gnu/libgl2ps.so
vio: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
vio: /usr/lib/x86_64-linux-gnu/libtheoradec.so
vio: /usr/lib/x86_64-linux-gnu/libogg.so
vio: /usr/lib/x86_64-linux-gnu/libxml2.so
vio: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_io.so
vio: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
vio: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_search.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_features.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
vio: /usr/lib/x86_64-linux-gnu/libqhull.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_people.so
vio: /usr/lib/x86_64-linux-gnu/libboost_system.so
vio: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
vio: /usr/lib/x86_64-linux-gnu/libboost_thread.so
vio: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
vio: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
vio: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
vio: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
vio: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
vio: /usr/lib/x86_64-linux-gnu/libboost_regex.so
vio: /usr/lib/x86_64-linux-gnu/libqhull.so
vio: /usr/lib/libOpenNI.so
vio: /usr/lib/libOpenNI2.so
vio: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
vio: /usr/lib/x86_64-linux-gnu/libfreetype.so
vio: /usr/lib/x86_64-linux-gnu/libz.so
vio: /usr/lib/x86_64-linux-gnu/libvtkDomainsChemistry-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libexpat.so
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneric-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersHyperTree-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelFlowPaths-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelGeometry-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelImaging-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelMPI-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelStatistics-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersProgrammable-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersPython-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libpython2.7.so
vio: /usr/lib/libvtkWrappingTools-6.3.a
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersReebGraph-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersSMP-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersSelection-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersVerdict-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkverdict-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libjpeg.so
vio: /usr/lib/x86_64-linux-gnu/libpng.so
vio: /usr/lib/x86_64-linux-gnu/libtiff.so
vio: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtOpenGL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtSQL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQtWebkit-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkViewsQt-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libproj.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOAMR-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
vio: /usr/lib/x86_64-linux-gnu/libsz.so
vio: /usr/lib/x86_64-linux-gnu/libdl.so
vio: /usr/lib/x86_64-linux-gnu/libm.so
vio: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOEnSight-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
vio: /usr/lib/x86_64-linux-gnu/libnetcdf.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOExport-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingGL2PS-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libgl2ps.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOFFMPEG-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOMovie-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
vio: /usr/lib/x86_64-linux-gnu/libtheoradec.so
vio: /usr/lib/x86_64-linux-gnu/libogg.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOGDAL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOGeoJSON-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOImport-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOInfovis-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libxml2.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOMINC-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOMPIImage-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOMPIParallel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOParallel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIONetCDF-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOMySQL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOODBC-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOParallelExodus-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOParallelLSDyna-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOParallelNetCDF-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOParallelXML-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOPostgreSQL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOVPIC-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkVPIC-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOVideo-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOXdmf2-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkxdmf2-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingMath-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingMorphological-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingStatistics-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingStencil-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkLocalExample-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI4Py-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingExternal-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingImage-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingMatplotlib-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallelLIC-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeAMR-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeOpenGL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkTestingGenericBridge-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkTestingIOSQL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkTestingRendering-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkViewsGeovis-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkWrappingJava-6.3.so.6.3.0
vio: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
vio: /usr/local/lib/libopencv_dnn.so.4.6.0
vio: /usr/local/lib/libopencv_calib3d.so.4.6.0
vio: /usr/local/lib/libopencv_features2d.so.4.6.0
vio: /usr/local/lib/libopencv_flann.so.4.6.0
vio: /usr/local/lib/libopencv_imgproc.so.4.6.0
vio: /usr/local/lib/libopencv_core.so.4.6.0
vio: /usr/lib/x86_64-linux-gnu/libpcl_common.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_io.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_search.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_features.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
vio: /usr/lib/x86_64-linux-gnu/libpcl_people.so
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersFlowPaths-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
vio: /usr/lib/x86_64-linux-gnu/libtheoradec.so
vio: /usr/lib/x86_64-linux-gnu/libogg.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOExodus-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkexoIIc-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
vio: /usr/lib/x86_64-linux-gnu/libnetcdf.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOLSDyna-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libxml2.so
vio: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
vio: /usr/lib/x86_64-linux-gnu/libsz.so
vio: /usr/lib/x86_64-linux-gnu/libdl.so
vio: /usr/lib/x86_64-linux-gnu/libm.so
vio: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
vio: /usr/lib/x86_64-linux-gnu/libvtkWrappingPython27Core-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkPythonInterpreter-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libpython2.7.so
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingLIC-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.9.5
vio: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.9.5
vio: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.9.5
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersAMR-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkParallelCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libGLU.so
vio: /usr/lib/x86_64-linux-gnu/libSM.so
vio: /usr/lib/x86_64-linux-gnu/libICE.so
vio: /usr/lib/x86_64-linux-gnu/libX11.so
vio: /usr/lib/x86_64-linux-gnu/libXext.so
vio: /usr/lib/x86_64-linux-gnu/libXt.so
vio: /usr/lib/x86_64-linux-gnu/libvtkIOSQL-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkViewsInfovis-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersImaging-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkGeovisCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOXML-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkInfovisLayout-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkInfovisBoostGraphAlgorithms-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libz.so
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libfreetype.so
vio: /usr/lib/x86_64-linux-gnu/libGL.so
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libvtksys-6.3.so.6.3.0
vio: /usr/lib/x86_64-linux-gnu/libproj.so
vio: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0
vio: CMakeFiles/vio.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sf/Desktop/vio/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vio"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vio.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vio.dir/build: vio
.PHONY : CMakeFiles/vio.dir/build

CMakeFiles/vio.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vio.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vio.dir/clean

CMakeFiles/vio.dir/depend:
	cd /home/sf/Desktop/vio/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sf/Desktop/vio /home/sf/Desktop/vio /home/sf/Desktop/vio/build /home/sf/Desktop/vio/build /home/sf/Desktop/vio/build/CMakeFiles/vio.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vio.dir/depend

