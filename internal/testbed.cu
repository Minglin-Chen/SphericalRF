#include <filesystem/path.h>

#include "internal/testbed.h"
#include "internal/dataset/nerf_synthetic.h"
#include "internal/dataset/mipnerf360.h"
#include "internal/dataset/lightfield.h"
#include "internal/dataset/tanksandtemples.h"
#include "internal/utils/common_device.h"
#include "internal/utils/adam_optimizer.h"
#include "internal/utils/json_binding.h"

using namespace Eigen;
using namespace tcnn;


Testbed::Testbed() 
{
	// CUDA Version
	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later."};
	}

	// NVIDIA GPU Hardware
	cudaDeviceProp props;
	cudaError_t error = cudaGetDeviceProperties(&props, 0);
	if (error != cudaSuccess) {
		throw std::runtime_error{std::string{"cudaGetDeviceProperties() returned an error: "} + cudaGetErrorString(error)};
	}
	if (!((props.major * 10 + props.minor) >= 75)) {
		throw std::runtime_error{"Turing Tensor Core operations must be run on a machine with compute capability at least 75."};
	}

	// CUDA Stream
	CUDA_CHECK_THROW(cudaStreamCreate(&m_stream));
}

Testbed::~Testbed() {}

/******************************************************************************************
 * Configuration & I/O
 ******************************************************************************************/
void Testbed::load_model_config(const std::string& config_path)
{
	// 0. parse config file
	const filesystem::path _config_path = config_path;
	if (_config_path.empty() || !_config_path.exists()) {
		tlog::error() << "The model config must exist, but got: " << _config_path.str();
		throw std::runtime_error{"The model config must exist."};
	}
	tlog::info() << "Load model config from: " << _config_path.str();

	if (tcnn::equals_case_insensitive(_config_path.extension(), "json")) {
		std::ifstream f{ _config_path.str() };
		m_model_config = json::parse(f, nullptr, true, true);
	} else if (equals_case_insensitive(_config_path.extension(), "msgpack")) {
		std::ifstream f{_config_path.str(), std::ios::in | std::ios::binary};
		m_model_config = json::from_msgpack(f);
	} else {
		tlog::error() << "The model config must be a .json or .msgpack file, but got: " << _config_path.str();
		throw std::runtime_error{"The model config must be a .json or .msgpack file."};
	}

	// model configuration
	// - sampler
	m_sampler = std::make_shared<OccupancySampler>(m_model_config["samlper"]);
	// - network
	m_network = std::make_shared<NGPNetwork<precision_t>>(m_model_config["network"]);
	// - render
	m_render = std::make_shared<RayMarcher>(m_model_config["render"]);
	// - loss, optimizer, and trainer
	m_loss_type = string_to_loss_type(m_model_config["loss"]);
	// Some of the Nerf-supported losses are not supported by tcnn::Loss,
	// so just create a dummy L2 loss there. The NeRF code path will bypass
	// the tcnn::Loss in any case.
	m_loss.reset(create_loss<precision_t>({{"otype", "L2"}}));
	m_optimizer.reset(create_optimizer<precision_t>(m_model_config["optimizer"]));
	m_trainer = std::make_shared<Trainer<float, precision_t, precision_t>>(m_network, m_optimizer, m_loss, m_seed);

	m_training_buffer.i_step = 0;
}

void Testbed::load_training_data(
	const std::string& data_path, 
	const float scale, 
	const Vector3f offset,
	const bool discard_saturation,
	const std::string& dataset_type) 
{
	// 0. parse config file
	const filesystem::path _data_path = data_path;
	if (_data_path.empty() || !_data_path.exists()) {
		tlog::error() << "The data path must exist, but got: " << _data_path.str();
		throw std::runtime_error{"The data path must exist."};
	}
	if (!tcnn::equals_case_insensitive(_data_path.extension(), "json")) {
		tlog::error() << "The data path must be a .json file, but got: " << _data_path.str();
		throw std::runtime_error{"The data path must be a .json file."};
	}

	// 1. load data
	if (dataset_type == "nerf_synthetic") {
		m_training_buffer.dataset = std::make_shared<NeRFSynthetic>();
	} else if (dataset_type == "mipnerf360") {
		m_training_buffer.dataset = std::make_shared<MipNeRF360>();
	} else if (dataset_type == "lightfield") {
		m_training_buffer.dataset = std::make_shared<LightField>();
	} else if (dataset_type == "tanksandtemples") {
		m_training_buffer.dataset = std::make_shared<TanksAndTemples>();
	} else {
		throw std::runtime_error{"Unknown dataset type."};
	}
	m_training_buffer.dataset->load_from_json(_data_path.str(), scale, offset, discard_saturation);

	// 2. other settings
	m_scene_scale = m_training_buffer.dataset->scale;
	m_scene_offset = m_training_buffer.dataset->offset;
	tlog::info() << "Scene scale: " << m_scene_scale;
	tlog::info() << "Scene offset: " << "(" 
				<< m_scene_offset.x() << ", "
				<< m_scene_offset.y() << ", "
				<< m_scene_offset.z() << ")";

	m_sampler->init(
		m_training_buffer.dataset->n_images,
		m_training_buffer.dataset->image_resolution,
		m_training_buffer.dataset->focal_length,
		m_training_buffer.dataset->principal_point,
		m_training_buffer.dataset->xforms,
		m_training_buffer.dataset->images,
		m_rng,
		m_stream
	);
}

void Testbed::save_snapshot(const std::string& snapshot_path, bool serialize_optimizer) 
{
	// network & (optional) optimizer
	m_model_config["snapshot"] = m_trainer->serialize(serialize_optimizer);
	// sampler
	m_model_config["snapshot"]["density_grid"] = m_sampler->density_grid;

	// others
	m_model_config["snapshot"]["scene_scale"] = m_scene_scale;
	to_json(m_model_config["snapshot"]["scene_offset"], m_scene_offset);

	// write
	std::ofstream f(snapshot_path, std::ios::out | std::ios::binary);
	json::to_msgpack(m_model_config, f);
}

void Testbed::load_snapshot(const std::string& snapshot_path) 
{
	// load
	load_model_config(snapshot_path);
	tlog::info() << "load snapshot from: " << snapshot_path;
	if (!m_model_config.contains("snapshot")) {
		tlog::error() << "ERROR in loading snapshot.";
		throw std::runtime_error{"ERROR in loading snapshot."};
	}

	// others
	tlog::info() << "load scene info";
	m_scene_scale = m_model_config["snapshot"]["scene_scale"];
	from_json(m_model_config["snapshot"]["scene_offset"], m_scene_offset);
	
	// sampler
	m_sampler->density_grid = m_model_config["snapshot"]["density_grid"];
	m_sampler->update_grid_bitfield(m_stream);
	// network & (optional) optimizer
	m_trainer->deserialize(m_model_config["snapshot"]);
}

/******************************************************************************************
 * Training
 ******************************************************************************************/
void Testbed::train(uint32_t n_training_steps, uint32_t target_batch_size) 
{
	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_buffer.training_prep_ms = std::chrono::duration<float, std::milli>(
				std::chrono::steady_clock::now()-start).count();
		}};
		
		if (m_sampler->use_grid) {
			train_prep_nerf(target_batch_size, n_training_steps, m_stream);
		}
		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));
	}

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_buffer.training_ms = std::chrono::duration<float, std::milli>(
				std::chrono::steady_clock::now()-start).count();
		}};

		train_nerf(target_batch_size, n_training_steps, m_stream);
		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));
	}
}

void Testbed::train_prep_nerf(uint32_t batch_size, uint32_t n_training_steps, cudaStream_t stream)
{
	const uint32_t n_total_elements 		= m_sampler->n_total_elements;
	const uint32_t n_uniform_positions 		= m_training_buffer.i_step >= 256 ? n_total_elements/4 : n_total_elements;
	const uint32_t n_nonuniform_positions 	= m_training_buffer.i_step >= 256 ? n_total_elements/4 : 0;
	const uint32_t n_total_positions 		= n_uniform_positions + n_nonuniform_positions;

	// sample positions
	m_sampler->sample_positions_from_grid(
		n_uniform_positions,
		n_nonuniform_positions,
		m_training_buffer.grid_positions,
		m_training_buffer.grid_indices,
		stream
	);

	// inference density
	const uint32_t padded_output_width = m_network->padded_density_output_width();
	m_training_buffer.rgbsigma.enlarge(padded_output_width * n_total_positions);
	GPUMatrix<precision_t> rgbsigma_matrix(m_training_buffer.rgbsigma.data(), padded_output_width, n_total_positions);
	m_network->density(
		stream, 
		{(float*)m_training_buffer.grid_positions.data(), sizeof(Position)/sizeof(float)}, 
		rgbsigma_matrix, 
		false);

	// update grid
	m_sampler->update_grid(
		n_total_positions,
		padded_output_width,
		m_training_buffer.rgbsigma,
		m_training_buffer.grid_indices,
		m_density_activation,
		m_stream
	);
	m_sampler->update_grid_bitfield(stream);
}

void Testbed::train_nerf(uint32_t target_batch_size, uint32_t n_training_steps, cudaStream_t stream)
{
	uint32_t& n_rays = m_training_buffer.n_rays_per_batch;

	// allocation
	m_training_buffer.n_rays_counter.enlarge(n_training_steps);
	uint32_t* n_rays_counter = m_training_buffer.n_rays_counter.data();
	CUDA_CHECK_THROW(cudaMemsetAsync(n_rays_counter, 0, sizeof(uint32_t)*n_training_steps, stream));

	m_training_buffer.n_coords_counter.enlarge(n_training_steps);
	uint32_t* n_coords_counter = m_training_buffer.n_coords_counter.data();
	CUDA_CHECK_THROW(cudaMemsetAsync(n_coords_counter, 0, sizeof(uint32_t)*n_training_steps, stream));

	m_training_buffer.n_compacted_coords_counter.enlarge(n_training_steps);
	uint32_t* n_compacted_coords_counter = m_training_buffer.n_compacted_coords_counter.data();
	CUDA_CHECK_THROW(cudaMemsetAsync(n_compacted_coords_counter, 0, sizeof(uint32_t)*n_training_steps, stream));

	m_training_buffer.loss.enlarge(n_rays * n_training_steps);
	float* loss = m_training_buffer.loss.data();
	CUDA_CHECK_THROW(cudaMemsetAsync(loss, 0, sizeof(float)*n_rays*n_training_steps, stream));
	
	// training
	for (uint32_t i = 0; i < n_training_steps; ++i) {
		train_nerf_step(
			target_batch_size, 
			n_rays, 
			n_rays_counter + i, 
			n_coords_counter + i, 
			n_compacted_coords_counter + i, 
			loss + n_rays*i, 
			m_stream);
	}

	// GPU -> CPU
	auto& n_rays_counter_cpu 				= m_training_buffer.n_rays_counter_cpu;
	auto& n_coords_counter_cpu 				= m_training_buffer.n_coords_counter_cpu;
	auto& n_compacted_coords_counter_cpu 	= m_training_buffer.n_compacted_coords_counter_cpu;
	n_rays_counter_cpu.resize(n_training_steps);
	n_coords_counter_cpu.resize(n_training_steps);
	n_compacted_coords_counter_cpu.resize(n_training_steps);
	m_training_buffer.n_rays_counter.copy_to_host(n_rays_counter_cpu, n_training_steps);
	m_training_buffer.n_coords_counter.copy_to_host(n_coords_counter_cpu, n_training_steps);
	m_training_buffer.n_compacted_coords_counter.copy_to_host(n_compacted_coords_counter_cpu, n_training_steps);

	// measurement
	m_training_buffer.measured_batch_size_before_compaction = 0;
	m_training_buffer.measured_batch_size = 0;
	for (uint32_t i = 0; i < n_training_steps; ++i) {
		if (n_coords_counter_cpu[i] == 0 || n_compacted_coords_counter_cpu[i] == 0) {
			tlog::error() << "Training generated 0 samples. Aborting training.";
			throw std::runtime_error{"Training generated 0 samples. Aborting training."};
		}
		m_training_buffer.measured_batch_size_before_compaction += n_coords_counter_cpu[i];
		m_training_buffer.measured_batch_size += n_compacted_coords_counter_cpu[i];
	}
	m_training_buffer.measured_batch_size_before_compaction /= n_training_steps;
	m_training_buffer.measured_batch_size /= n_training_steps;

	// statistic loss
	m_training_buffer.loss_scalar = reduce_sum(loss, n_rays*n_training_steps, stream) / (float)(n_training_steps);
	m_training_buffer.loss_scalar *= (float)m_training_buffer.measured_batch_size / (float)target_batch_size;

	// dynamically determine the number of rays for each step
	n_rays = (uint32_t)((float)n_rays * (float)target_batch_size / (float)m_training_buffer.measured_batch_size);
	n_rays = std::min(next_multiple(n_rays, BATCH_SIZE_MULTIPLE), 1u << 18);
}

void Testbed::train_nerf_step(
	uint32_t 		target_batch_size, 
	uint32_t 		n_rays_per_batch, 
	uint32_t* 		ray_counter,
	uint32_t* 		coord_counter, 
	uint32_t* 		compacted_coord_counter, 
	float* 			loss, 
	cudaStream_t 	stream
) {
	const uint32_t max_samples = target_batch_size * 16;
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_inference = next_multiple(
			std::min(m_training_buffer.measured_batch_size_before_compaction, max_samples), BATCH_SIZE_MULTIPLE);

	// 1. for each ray in the training step, sample points along the ray
	// record the rays
	m_training_buffer.rays.enlarge(n_rays_per_batch);
	m_training_buffer.rays_rgba.enlarge(n_rays_per_batch);
	// number of steps each ray took, and the first offset of samples
	m_training_buffer.ray_coord_indices.enlarge(n_rays_per_batch*2);

	// space for sampled points (coords = positions + directions)
	m_training_buffer.coords.enlarge(max_samples);
	m_training_buffer.coords_compacted.enlarge(target_batch_size);
	// space for network output
	m_training_buffer.rgbsigma.enlarge(max_samples * padded_output_width); 
	// space for loss gradients w.r.t. mlp output
	m_training_buffer.dloss_drgbsigma.enlarge(target_batch_size * padded_output_width); 

	// record the number of sampled rays used to train since now
	if (m_training_buffer.i_step == 0) { m_training_buffer.n_rays_total = 0; }
	uint32_t n_rays_total = m_training_buffer.n_rays_total;
	m_training_buffer.n_rays_total += n_rays_per_batch;

	m_sampler->sample_coords_from_dataset(
		n_rays_per_batch,
		n_rays_total,
		max_inference,
		m_training_buffer.dataset,
		m_training_buffer.rays,
		m_training_buffer.rays_rgba,
		ray_counter,
		m_training_buffer.ray_coord_indices,
		m_training_buffer.coords,
		coord_counter,
		m_rng,
		stream
	);

	// 2. network forward
	GPUMatrix<float> coords_matrix((float*)m_training_buffer.coords.data(), sizeof(Coordinate)/sizeof(float), max_inference);
	GPUMatrix<precision_t> rgbsigma_matrix(m_training_buffer.rgbsigma.data(), padded_output_width, max_inference);
	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

	// 3. compute the loss and gradients
	m_render->volume_rendering_with_loss_and_gradient(
		n_rays_per_batch,
		target_batch_size,
		// network input (ray & coordinate)
		m_training_buffer.rays,
		m_training_buffer.rays_rgba,
		ray_counter,
		m_training_buffer.ray_coord_indices,
		m_training_buffer.coords,
		m_training_buffer.coords_compacted,
		compacted_coord_counter,
		// network output (density & color) with gradient
		m_training_buffer.rgbsigma,
		m_training_buffer.dloss_drgbsigma,
		padded_output_width,
		m_rgb_activation,
		m_density_activation,
		// loss
		loss,
		m_loss_type,
		LOSS_SCALE,
		// sampling parameters
		m_sampler->aabb,
		m_sampler->n_cascades,
		m_sampler->min_cone_stepsize,
		m_sampler->density_grid_mean,
		m_sampler->min_optical_thickness,
		m_sampler->use_grid,
		// others
		m_background_color.head<3>(),
		m_color_space,
		m_rng,
		stream
	);

	fill_rollover_and_rescale<network_precision_t>
		<<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, compacted_coord_counter, m_training_buffer.dloss_drgbsigma.data()
	);
	fill_rollover<Coordinate><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, compacted_coord_counter, m_training_buffer.coords_compacted.data()
	);

	// compute the gradient for network
	GPUMatrix<float> compacted_coords_matrix((float*)m_training_buffer.coords_compacted.data(), sizeof(Coordinate)/sizeof(float), target_batch_size);
	GPUMatrix<precision_t> compacted_rgbsigma_matrix(m_training_buffer.rgbsigma.data(), padded_output_width, target_batch_size);
	GPUMatrix<precision_t> gradient_matrix(m_training_buffer.dloss_drgbsigma.data(), padded_output_width, target_batch_size);
	m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix);
	m_network->backward(stream, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix);

	// update the network parameters
	m_rng.advance();
	m_trainer->optimizer_step(stream, LOSS_SCALE);

	++m_training_buffer.i_step;
}

/******************************************************************************************
 * Rendering
 ******************************************************************************************/
__global__ void compact_rays_kernel(
	const uint32_t n_elements,
	RayPayload* src_payloads,	Array4f* src_rgba,
	RayPayload* dst_payloads,	Array4f* dst_rgba,
	RayPayload* final_payloads,	Array4f* final_rgba,
	uint32_t* n_counter, 
	uint32_t* n_final_counter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	RayPayload& 	payload = src_payloads[i];
	Array4f&		rgba 	= src_rgba[i];

	if (payload.alive) {
		uint32_t idx		= atomicAdd(n_counter, 1);
		dst_payloads[idx]	= payload;
		dst_rgba[idx]		= rgba;
	} else if (rgba.w() > 0.001f) {
		uint32_t idx 		= atomicAdd(n_final_counter, 1);
		final_payloads[idx]	= payload;
		final_rgba[idx] 	= rgba;
	}
}

__global__ void shade_kernel(
	const uint32_t n_rays, 
	const Array4f* __restrict__ rgba, 
	const RayPayload* __restrict__ payloads, 
	Array4f* __restrict__ frame_buffer,
	const ERenderMode render_mode, 
	const bool train_in_linear_colors
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	const RayPayload& payload = payloads[i];

	Array4f tmp = rgba[i];

	if (render_mode == ERenderMode::Normals) {
		Array3f n = tmp.head<3>().matrix().normalized().array();
		tmp.head<3>() = (0.5f * n + Array3f::Constant(0.5f)) * tmp.w();
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade)) {
		// Accumulate in linear colors
		tmp.head<3>() = srgb_to_linear(tmp.head<3>());
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.w());
}

void Testbed::render_frame(const Matrix<float, 3, 4>& camera_matrix, CudaRenderBuffer& render_buffer, bool to_srgb) 
{
	const Vector2i 		resolution 		= render_buffer.resolution();
	const Vector2f		relative_fl 	= m_rendering_buffer.relative_focal_length;
	const Vector2f 		focal_length 	= Vector2f{resolution.x() * relative_fl.x(), resolution.y() * relative_fl.y()};
	const Vector2f 		principal_point = m_rendering_buffer.principal_point;
	const uint32_t 		n_pixels 		= resolution.x() * resolution.y();
	const ERenderMode 	render_mode 	= m_rendering_buffer.render_mode;

	// Memory Allocation
	{
		const uint32_t n_elements = next_multiple(n_pixels, BATCH_SIZE_MULTIPLE);

		m_rendering_buffer.rays[0].enlarge(n_elements);
		m_rendering_buffer.rays[1].enlarge(n_elements);
		m_rendering_buffer.rays_hit.enlarge(n_elements);

		m_rendering_buffer.coords.enlarge(n_elements * m_rendering_buffer.MAX_STEPS_INBETWEEN_COMPACTION);
		m_rendering_buffer.rgbsigma.enlarge(n_elements * m_rendering_buffer.MAX_STEPS_INBETWEEN_COMPACTION * m_network->padded_output_width());
	}

	// Clear Buffer & Generate Rays 
	{
		render_buffer.clear_frame_buffer(m_stream);

		m_sampler->generate_rays_from_camera_matrix(
			resolution,
			focal_length,
			principal_point,
			camera_matrix,
			m_rendering_buffer.rays[0].payload,
			render_buffer,
			m_stream
		);

		CUDA_CHECK_THROW(cudaMemsetAsync(m_rendering_buffer.rays[0].rgba.data(), 0, m_rendering_buffer.rays[0].rgba.get_bytes(), m_stream));
	}
	
	// Render each ray
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rendering_buffer.n_hit_counter.data(), 0, sizeof(uint32_t), m_stream));

	uint32_t i_steps_offset = 1;
	uint32_t double_buffer_index = 0;
	uint32_t n_alive = n_pixels;
	while (i_steps_offset < MARCH_ITER)
	{
		// 0. collect alive rays
		RaysSoA& rays_previous	= m_rendering_buffer.rays[ double_buffer_index    % 2];
		RaysSoA& rays_current 	= m_rendering_buffer.rays[(double_buffer_index+1) % 2];
		++double_buffer_index;

		CUDA_CHECK_THROW(cudaMemsetAsync(m_rendering_buffer.n_alive_counter.data(), 0, sizeof(uint32_t), m_stream));
		linear_kernel(compact_rays_kernel, 0, m_stream,
			n_alive,
			rays_previous.payload.data(), 				rays_previous.rgba.data(), 
			rays_current.payload.data(), 				rays_current.rgba.data(), 
			m_rendering_buffer.rays_hit.payload.data(), m_rendering_buffer.rays_hit.rgba.data(), 
			m_rendering_buffer.n_alive_counter.data(),
			m_rendering_buffer.n_hit_counter.data()
		);
		CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_rendering_buffer.n_alive_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, m_stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));

		if (n_alive == 0) break;

		// 1. generate rays
		uint32_t n_steps = tcnn::clamp(
			n_pixels / n_alive, 
			m_rendering_buffer.MIN_STEPS_INBETWEEN_COMPACTION, 
			m_rendering_buffer.MAX_STEPS_INBETWEEN_COMPACTION);
		m_sampler->generate_coords_from_rays(n_alive, n_steps, rays_current.payload, m_rendering_buffer.coords, m_stream);

		// 2. network inference
		uint32_t n_elements = next_multiple(n_alive * n_steps, BATCH_SIZE_MULTIPLE);
		GPUMatrix<float> coords_matrix((float*)m_rendering_buffer.coords.data(), sizeof(Coordinate)/sizeof(float), n_elements);
		GPUMatrix<precision_t> rgbsigma_matrix(m_rendering_buffer.rgbsigma.data(), m_network->padded_output_width(), n_elements);
		m_network->inference_mixed_precision(m_stream, coords_matrix, rgbsigma_matrix);

		if (render_mode == ERenderMode::Normals) {
			m_network->input_gradient(m_stream, 3, coords_matrix, coords_matrix);
		}

		// 3. rendering
		m_render->volume_rendering(
			n_alive,
			n_steps,
			i_steps_offset,
			rays_current,
			m_rendering_buffer.coords,
			m_rendering_buffer.rgbsigma,
			m_network->padded_output_width(),
			m_density_activation,
			m_rgb_activation,
			render_mode,
			m_sampler->warping_type,
			m_sampler->aabb,
			m_sampler->n_cascades,
			m_sampler->min_cone_stepsize,
			camera_matrix,
			1.f / m_scene_scale,
			m_stream
		);

		i_steps_offset += n_steps;
	}

	uint32_t n_hit = 0;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_rendering_buffer.n_hit_counter.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, m_stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));

	// Write to buffer
	RaysSoA& rays_hit = m_rendering_buffer.rays_hit;
	linear_kernel(shade_kernel, 0, m_stream,
		n_hit,
		rays_hit.rgba.data(),
		rays_hit.payload.data(),
		render_buffer.frame_buffer(),	
		render_mode,
		m_render->train_in_linear_color
	);

	// Post-processing
	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_rendering_buffer.tonemap_curve);
	render_buffer.accumulate(m_stream);
	render_buffer.tonemap(m_rendering_buffer.exposure, m_background_color, to_srgb ? EColorSpace::SRGB : EColorSpace::Linear, m_stream);
	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream));
}