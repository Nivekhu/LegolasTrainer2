# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevin/Desktop/LegolasTrainer2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/Desktop/LegolasTrainer2

# Include any dependencies generated for this target.
include CMakeFiles/LegoTrain.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LegoTrain.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LegoTrain.dir/flags.make

CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o: CMakeFiles/LegoTrain.dir/flags.make
CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o: LegoTrain.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kevin/Desktop/LegolasTrainer2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o -c /home/kevin/Desktop/LegolasTrainer2/LegoTrain.cpp

CMakeFiles/LegoTrain.dir/LegoTrain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LegoTrain.dir/LegoTrain.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kevin/Desktop/LegolasTrainer2/LegoTrain.cpp > CMakeFiles/LegoTrain.dir/LegoTrain.cpp.i

CMakeFiles/LegoTrain.dir/LegoTrain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LegoTrain.dir/LegoTrain.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kevin/Desktop/LegolasTrainer2/LegoTrain.cpp -o CMakeFiles/LegoTrain.dir/LegoTrain.cpp.s

CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.requires:

.PHONY : CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.requires

CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.provides: CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.requires
	$(MAKE) -f CMakeFiles/LegoTrain.dir/build.make CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.provides.build
.PHONY : CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.provides

CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.provides.build: CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o


# Object files for target LegoTrain
LegoTrain_OBJECTS = \
"CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o"

# External object files for target LegoTrain
LegoTrain_EXTERNAL_OBJECTS =

LegoTrain: CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o
LegoTrain: CMakeFiles/LegoTrain.dir/build.make
LegoTrain: /usr/local/lib/libopencv_gapi.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_stitching.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_aruco.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_bgsegm.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_bioinspired.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_ccalib.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_dnn_objdetect.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_dpm.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_face.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_freetype.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_fuzzy.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_hfs.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_img_hash.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_line_descriptor.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_reg.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_rgbd.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_saliency.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_stereo.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_structured_light.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_superres.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_surface_matching.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_tracking.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_videostab.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_xfeatures2d.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_xobjdetect.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_xphoto.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_shape.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_phase_unwrapping.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_optflow.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_ximgproc.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_datasets.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_plot.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_text.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_dnn.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_ml.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_video.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_objdetect.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_calib3d.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_features2d.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_flann.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_highgui.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_videoio.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_imgcodecs.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_photo.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_imgproc.so.4.0.1
LegoTrain: /usr/local/lib/libopencv_core.so.4.0.1
LegoTrain: CMakeFiles/LegoTrain.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kevin/Desktop/LegolasTrainer2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable LegoTrain"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LegoTrain.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LegoTrain.dir/build: LegoTrain

.PHONY : CMakeFiles/LegoTrain.dir/build

CMakeFiles/LegoTrain.dir/requires: CMakeFiles/LegoTrain.dir/LegoTrain.cpp.o.requires

.PHONY : CMakeFiles/LegoTrain.dir/requires

CMakeFiles/LegoTrain.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LegoTrain.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LegoTrain.dir/clean

CMakeFiles/LegoTrain.dir/depend:
	cd /home/kevin/Desktop/LegolasTrainer2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/Desktop/LegolasTrainer2 /home/kevin/Desktop/LegolasTrainer2 /home/kevin/Desktop/LegolasTrainer2 /home/kevin/Desktop/LegolasTrainer2 /home/kevin/Desktop/LegolasTrainer2/CMakeFiles/LegoTrain.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LegoTrain.dir/depend

