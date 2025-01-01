#pragma once

#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include "internal/dataset/dataset.h"
#include "internal/sampler/occupancy_sampler.h"
#include "internal/network/ngp_network.h"
#include "internal/render/ray_marcher.h"
#include "internal/utils/common.h"
#include "internal/utils/common_device.h"
#include "internal/utils/loss.h"
#include "internal/utils/trainable_buffer.h"
#include "internal/utils/render_buffer.h"

#ifdef NGP_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif


class Testbed {
public:
	Testbed();
	~Testbed();
	
	/******************************************************************************************
	 * Configuration & I/O
	 ******************************************************************************************/
	// model
	void load_model_config(const std::string& config_path);
	void save_snapshot(const std::string& snapshot_path, bool serialize_optimizer);
	void load_snapshot(const std::string& snapshot_path);
	
	// dataset
	void load_training_data(
		const std::string& data_path, 
		const float scale, 
		const Eigen::Vector3f offset,
		const bool discard_saturation,
		const std::string& dataset_type);

	/******************************************************************************************
	 * Training
	 ******************************************************************************************/
	void train(uint32_t n_training_steps, uint32_t batch_size);

	void train_prep_nerf(uint32_t batch_size, uint32_t n_training_steps, cudaStream_t stream);

	void train_nerf(uint32_t target_batch_size, uint32_t n_training_steps, cudaStream_t stream);
	void train_nerf_step(
		uint32_t target_batch_size, 
		uint32_t n_rays_per_batch, 
		uint32_t* ray_counter,
		uint32_t* coord_counter, 
		uint32_t* compacted_coord_counter, 
		float* loss, 
		cudaStream_t stream);
	
	/******************************************************************************************
	 * Rendering
	 ******************************************************************************************/
#ifdef NGP_PYTHON
	pybind11::array_t<float> render(int width, int height, int spp, bool to_srgb);
#endif
	void render_frame(
		const Eigen::Matrix<float, 3, 4>& camera_matrix, 
		CudaRenderBuffer& render_buffer, 
		bool to_srgb = true);

	float fov() const { 
		return focal_length_to_fov(1.0f, m_rendering_buffer.relative_focal_length[m_rendering_buffer.fov_axis]); 
	}
	void set_fov(float val) { 
		m_rendering_buffer.relative_focal_length = Eigen::Vector2f::Constant(fov_to_focal_length(1, val)); 
	}
	Eigen::Vector2f fov_xy() const { 
		return focal_length_to_fov(Eigen::Vector2i::Ones(), m_rendering_buffer.relative_focal_length); 
	}
	void set_fov_xy(const Eigen::Vector2f& val) { 
		m_rendering_buffer.relative_focal_length = fov_to_focal_length(Eigen::Vector2i::Ones(), val); 
	}

	void set_nerf_camera_matrix(const Eigen::Matrix<float, 3, 4>& cam) {
		m_rendering_buffer.camera_matrix = spec_opengl_to_opencv(cam, m_scene_scale, m_scene_offset);
	}

public:
	uint32_t 				m_seed = 43;
	default_rng_t 			m_rng = default_rng_t{(uint64_t)m_seed};

	// CUDA stuff
	cudaStream_t 			m_stream;

	/******************************************************************************************
	 * Model
	 ******************************************************************************************/
	nlohmann::json														m_model_config;

    // sampler
    std::shared_ptr<OccupancySampler>                           		m_sampler;
    // network
	std::shared_ptr<NGPNetwork<precision_t>> 							m_network;
    // render
    std::shared_ptr<RayMarcher>                                     	m_render;
	// loss, optimizer & trainer
	ELossType 															m_loss_type;
	std::shared_ptr<tcnn::Loss<precision_t>> 							m_loss;
	std::shared_ptr<tcnn::Optimizer<precision_t>> 						m_optimizer;
	std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> 	m_trainer;
	// others
	EActivation 														m_density_activation = EActivation::Exponential;
	EActivation 														m_rgb_activation = EActivation::Logistic;
	Eigen::Array4f 														m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};
	EColorSpace 														m_color_space = EColorSpace::Linear;

	/******************************************************************************************
	 * Scene
	 ******************************************************************************************/
	float																m_scene_scale;
	Eigen::Vector3f														m_scene_offset;

	/******************************************************************************************
	 * Buffers used in training & rendering
	 ******************************************************************************************/
	struct TrainingBuffer {
		std::shared_ptr<Dataset> 			dataset;

		uint32_t 							n_rays_per_batch = 1<<12;
		uint32_t 							n_rays_total = 0;

		// space for occupancy grid updates
		tcnn::GPUMemory<Position> 			grid_positions;
		tcnn::GPUMemory<uint32_t> 			grid_indices;
		// space for rays
		tcnn::GPUMemory<Ray> 				rays;
		tcnn::GPUMemory<Eigen::Array4f>		rays_rgba;
		tcnn::GPUMemory<uint32_t> 			ray_counter;
		tcnn::GPUMemory<uint32_t> 			ray_coord_indices;
		// space for network input (i.e, coordinates)
		tcnn::GPUMemory<Coordinate> 		coords;
		tcnn::GPUMemory<Coordinate> 		coords_compacted;
		// space for network output and their gradient
		tcnn::GPUMemory<precision_t> 		rgbsigma;
		tcnn::GPUMemory<precision_t> 		dloss_drgbsigma;

		// in each training step, we sample rays, then sample points along these rays
		// spaces to record the number of sampled rays
		tcnn::GPUMemory<uint32_t> 	n_rays_counter;
		std::vector<uint32_t>		n_rays_counter_cpu;
		// spaces to record the number of sampled points in each training step
		tcnn::GPUMemory<uint32_t>	n_coords_counter;
		tcnn::GPUMemory<uint32_t>	n_compacted_coords_counter;
		std::vector<uint32_t>		n_coords_counter_cpu;
		std::vector<uint32_t>		n_compacted_coords_counter_cpu;
		uint32_t					measured_batch_size_before_compaction = std::numeric_limits<uint32_t>::max();
		uint32_t					measured_batch_size = std::numeric_limits<uint32_t>::max();

		// space to record the loss for each ray
		tcnn::GPUMemory<float> 		loss;
		float 						loss_scalar;

		// timing
		float 						training_prep_ms = 0;
		float 						training_ms = 0;

		uint32_t 					i_step = 0;
    } m_training_buffer;

	struct RenderingBuffer {
		ERenderMode 				render_mode = ERenderMode::Shade;
		ETonemapCurve 				tonemap_curve = ETonemapCurve::Identity;
		float 						exposure = 0.f;
		CudaRenderBuffer			render_buffer{std::make_shared<CudaSurface2D>()};

		// camera parameters
		uint32_t 					fov_axis = 1;
		Eigen::Vector2f 			relative_focal_length = Eigen::Vector2f::Ones();
		Eigen::Vector2f				principal_point = Eigen::Vector2f::Constant(0.5f);
		Eigen::Matrix<float, 3, 4> 	camera_matrix = Eigen::Matrix<float, 3, 4>::Zero();

		// space for rays
		RaysSoA 					rays[2];
		RaysSoA 					rays_hit;
		tcnn::GPUMemory<uint32_t>	n_hit_counter{1};
		tcnn::GPUMemory<uint32_t>	n_alive_counter{1};

		// space for network input & output
		tcnn::GPUMemory<Coordinate> 		coords;
		tcnn::GPUMemory<precision_t> 		rgbsigma;

		// others
		uint32_t 					MIN_STEPS_INBETWEEN_COMPACTION = 1;
		uint32_t 					MAX_STEPS_INBETWEEN_COMPACTION = 8;
	} m_rendering_buffer;
};