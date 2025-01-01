#pragma once

#include <vector>
#include <string>
#include <tiny-cuda-nn/gpu_memory.h>

#include "internal/utils/common.h"

#include "internal/dataset/dataset.h"


class LightField : public Dataset {

public:
	void load_from_json(
		const std::string& path,
		const float scale,
		const Eigen::Vector3f offset,
		const bool discard_saturation = false);
};