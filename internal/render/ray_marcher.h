#pragma once

#include <tiny-cuda-nn/gpu_memory.h>

#include <json/json.hpp>

#include "internal/sampler/bounding_box.h"
#include "internal/utils/common.h"
#include "internal/utils/loss.h"
#include "internal/utils/random_val.cuh"


class RayMarcher {
    using json = nlohmann::json;

public:
    RayMarcher(json& config);

	void volume_rendering_with_loss_and_gradient(
		const uint32_t n_rays,
		const uint32_t n_max_compacted_coords,
		// network input (ray & coordinate)
		const tcnn::GPUMemory<Ray>& rays, 
		const tcnn::GPUMemory<Eigen::Array4f>& rays_rgba,
		const uint32_t* __restrict__ n_rays_counter,
		tcnn::GPUMemory<uint32_t>& ray_coord_indices,
		const tcnn::GPUMemory<Coordinate>& coords,
		tcnn::GPUMemory<Coordinate>& compacted_coords,
		uint32_t* __restrict__ n_compacted_coords_counter,
		// network output (density & color) with gradient
		const tcnn::GPUMemory<precision_t> network_output,
		tcnn::GPUMemory<precision_t> dloss_doutput,
		const int padded_output_width,
		EActivation rgb_activation,
		EActivation density_activation,
		// loss
		float* __restrict__ loss_output,
		ELossType loss_type,
		float loss_scale,
		// sampling parameters
		BoundingBox aabb,
		const uint32_t n_cascades,
		const float min_cone_stepsize,
		const tcnn::GPUMemory<float> density_grid_mean,
		const float min_optical_thickness,
		const bool use_grid,
		// others
		Eigen::Array3f background_color,
		EColorSpace color_space,
		tcnn::default_rng_t rng,
		cudaStream_t stream
	);

    void volume_rendering(
		const uint32_t n_rays,
		const uint32_t n_steps,
		const uint32_t i_steps_offset,
		RaysSoA& rays,
		const tcnn::GPUMemory<Coordinate>& coords,
		const tcnn::GPUMemory<precision_t>& rgbsigma,
		const uint32_t padded_output_width,
		const EActivation density_activation,
		const EActivation rgb_activation,
		const ERenderMode render_mode,
		EWarpingType warping_type,
		const BoundingBox aabb,
		const uint32_t n_cascades,
		const float min_cone_stepsize,
		const Eigen::Matrix<float, 3, 4>& camera_matrix,
		const float depth_scale,
		cudaStream_t stream);

public:
	bool 						train_with_random_bg_color;
	bool						train_in_linear_color;
	float 						transmittance_threshold;

	ERegularizationType			regularization_type;
	// temporary space for distortion regularization
	tcnn::GPUMemory<float>      reg_distortion_buffer;
};