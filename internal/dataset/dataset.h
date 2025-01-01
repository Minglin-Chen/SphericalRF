#pragma once

#include <vector>
#include <string>
#include <tiny-cuda-nn/gpu_memory.h>

#include "internal/utils/common.h"


class Dataset {

public:
    virtual void load_from_json(
        const std::string& path,
		const float scale,
		const Eigen::Vector3f offset,
		const bool discard_saturation = false) {};

public:
    // scale & offset of the scene
    float                       scale = 0.33f;
    Eigen::Vector3f             offset = {0.5f, 0.5f, 0.5f};

	// total number of images in the dataset
	size_t 						n_images = 0;

	// image resolution
	Eigen::Vector2i 			image_resolution = {0, 0};
	// camera intrinsic parameters
	Eigen::Vector2f 			focal_length;
	Eigen::Vector2f 			principal_point = Eigen::Vector2f::Constant(0.5f);
	// camera poses
	tcnn::GPUMemory<Eigen::Matrix<float, 3, 4>> xforms;
	// image data
    tcnn::GPUMemory<__half>     images;
	// sampling probability
	tcnn::GPUMemory<__half>		sampling_probability;

	// filenames
    std::vector<std::string>    filenames;
};