#pragma once

#include "internal/utils/common.h"

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <json/json.hpp>


template <typename T>
__global__ void split_inputs_nerf(
	const uint32_t n_elements,
	const uint32_t n_partition_1_dims,
	const uint32_t n_partition_2_dims,
	const uint32_t stride,
	const T* __restrict__ inputs,
	T* __restrict__ partition_1,
	T* __restrict__ partition_2
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / stride;
	const uint32_t dim_idx = i - elem_idx * stride;

	T value = inputs[i];
	if (dim_idx < n_partition_1_dims) {
		partition_1[elem_idx * n_partition_1_dims + dim_idx] = value;
	} else {
		partition_2[elem_idx * n_partition_2_dims + dim_idx - n_partition_1_dims] = value;
	}
}

template <typename T>
__global__ void grab_density_network_output(
	const uint32_t n_elements,
	const uint32_t n_partition_1_dims,
	const uint32_t stride,
	const T* __restrict__ partition_1,
	T* __restrict__ inputs
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / n_partition_1_dims;
	const uint32_t dim_idx = i - elem_idx * n_partition_1_dims;

// #define RELU_NON_DENSITY
#ifdef RELU_NON_DENSITY
	// ReLU for non-density dims, i.e. dim_idx > 0
	inputs[elem_idx * stride + dim_idx] = (dim_idx == 0 || partition_1[i] > (T)0.0f) ? partition_1[i] : (T)0.0f;
#else
	// ReLU for non-density dims, i.e. dim_idx > 0
	inputs[elem_idx * stride + dim_idx] = partition_1[i];
#endif
}

template <typename T>
__global__ void assemble_rgbd(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgb_stride,
	const T* __restrict__ density,
	const T* __restrict__ rgb,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 4;
	const uint32_t dim_idx = i - elem_idx * 4;

	rgbd[i] = dim_idx < 3 ? rgb[elem_idx*rgb_stride + dim_idx] : density[elem_idx*density_stride];
}

template <typename T>
__global__ void extract_rgb(
	const uint32_t n_elements,
	const uint32_t rgb_stride,
	const T* __restrict__ rgbd,
	T* __restrict__ rgb
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	rgb[elem_idx*rgb_stride + dim_idx] = rgbd[elem_idx*4 + dim_idx];
}

template <typename T>
__global__ void extract_density_gradient(
	const uint32_t n_elements,
	const uint32_t width,
	const uint32_t stride,
	const T* __restrict__ rgbd,
	const T* __restrict__ rgb_in,
	const T* __restrict__ rgb_in_grad,
	T* __restrict__ density_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / width;
	const uint32_t dim_idx = i - elem_idx * width;

#ifdef RELU_NON_DENSITY
	// Backprop through ReLU if not the density dimension.
	float value = (dim_idx == 0 || rgb_in[i] > (T)0.0f) ? rgb_in_grad[elem_idx*stride + dim_idx] : (T)0.0f;
#else
	float value = rgb_in_grad[elem_idx*stride + dim_idx];
#endif
	if (dim_idx == 0) {
		value += (float)rgbd[elem_idx*4+3]; // Add density gradient
	}
	density_out[i] = (T)value;
}

template <typename T>
__global__ void extract_dir_gradient(
	const uint32_t n_elements,
	const uint32_t offset,
	const uint32_t width,
	const uint32_t stride,
	const T* __restrict__ rgb_in,
	T* __restrict__ density_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / width;
	const uint32_t dim_idx = i - elem_idx * width;

	density_out[i] = rgb_in[offset + elem_idx*stride + dim_idx];
}


template <typename T>
class NGPNetwork : public tcnn::Network<float, T> {
	using json = nlohmann::json;

public:
	NGPNetwork(json& config)
	{
		json& pos_encoding = config["encoding"];
		auto_determine_grid_parameters(pos_encoding);
		json& dir_encoding = config["dir_encoding"];
		json& density_network = config["network"];
		json& rgb_network = config["rgb_network"];

		uint32_t pos_encoding_alignment = get_encoding_alignment(density_network);
		m_pos_encoding.reset(tcnn::create_encoding<T>(m_n_pos_dims, pos_encoding, pos_encoding_alignment));

		uint32_t rgb_encoding_alignment = get_encoding_alignment(rgb_network);
		m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims, dir_encoding, rgb_encoding_alignment));

		json local_density_network_config = density_network;
		local_density_network_config["n_input_dims"] = m_pos_encoding->num_encoded_dims();
		if (!density_network.contains("n_output_dims")) {
			local_density_network_config["n_output_dims"] = 16;
		}
		m_density_network.reset(tcnn::create_network<T>(local_density_network_config));

		json local_rgb_network_config = rgb_network;
		m_rgb_network_input_width = tcnn::next_multiple(
			m_dir_encoding->num_encoded_dims() + m_density_network->padded_output_width() - 1, rgb_encoding_alignment);
		local_rgb_network_config["n_input_dims"] = m_rgb_network_input_width;
		local_rgb_network_config["n_output_dims"] = 3;
		m_rgb_network.reset(tcnn::create_network<T>(local_rgb_network_config));

		// info
		tlog::info()
			<< "Density model: " << m_n_pos_dims
			<< "--[" << std::string(pos_encoding["otype"])
			<< "]-->" << m_pos_encoding->num_encoded_dims()
			<< "--[" << std::string(density_network["otype"])
			<< "(neurons=" << (int)density_network["n_neurons"] << ",layers=" << ((int)density_network["n_hidden_layers"]+2) << ")"
			<< "]-->" << 1;
		tlog::info()
			<< "Color model:   " << m_n_dir_dims
			<< "--[" << std::string(dir_encoding["otype"])
			<< "]-->" << m_dir_encoding->num_encoded_dims() << "+" << m_density_network->padded_output_width()
			<< "--[" << std::string(rgb_network["otype"])
			<< "(neurons=" << (int)rgb_network["n_neurons"] << ",layers=" << ((int)rgb_network["n_hidden_layers"]+2) << ")"
			<< "]-->" << 3;
		tlog::info()
			<< "  total_encoding_params=" << m_pos_encoding->n_params() + m_dir_encoding->n_params()
			<< " total_network_params=" << this->n_params() - m_pos_encoding->n_params() - m_dir_encoding->n_params();
	}

	virtual ~NGPNetwork() {}

	void inference(cudaStream_t stream, const tcnn::GPUMatrix<float>& input, tcnn::GPUMatrix<float>& output) override {
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_density_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		inference_mixed_precision(stream, input, m_inference_network_output);
		tcnn::linear_kernel(tcnn::cast_from<T>, 0, stream, 
			m_inference_network_output.n_elements(), 
			m_inference_network_output.data(), 
			output.data());
	}

	void inference_mixed_precision(
		cudaStream_t stream, 
		const tcnn::GPUMatrix<float>& input, 
		tcnn::GPUMatrixDynamic<T>& output, 
		bool use_inference_matrices = true)
	{
		if (output.layout() != tcnn::CM) {
			throw std::runtime_error("NGPNetwork::inference_mixed_precision output must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_inference_density_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		// Perform positional encoding and directional encoding in parallel
		// Then perform density network query
		{
			tcnn::SyncedMultiStream synced_streams{stream, 2};

			m_pos_encoding->encode(
				synced_streams.get(0),
				batch_size,
				{input.data(), input.m()},
				{m_inference_density_network_input.data(), m_inference_density_network_input.m()},
				nullptr,
				use_inference_matrices
			);
			m_dir_encoding->encode(
				synced_streams.get(1),
				batch_size,
				{input.data() + m_dir_offset, input.m()},
				{m_inference_rgb_network_input.data() + m_inference_density_network_output.m(), m_inference_rgb_network_input.m()},
				nullptr,
				use_inference_matrices
			);

			m_density_network->inference_mixed_precision(
				synced_streams.get(0), 
				m_inference_density_network_input, 
				m_inference_density_network_output, 
				use_inference_matrices);

			tcnn::linear_kernel(grab_density_network_output<T>, 0, synced_streams.get(0),
				m_inference_density_network_output.n_elements(), 
				m_inference_density_network_output.m(), 
				m_inference_rgb_network_input.m(), /* stride */ 
				m_inference_density_network_output.data(), 
				m_inference_rgb_network_input.data()
			);
		}

		m_rgb_network->inference_mixed_precision(
			stream, 
			m_inference_rgb_network_input, 
			m_inference_rgb_network_output, 
			use_inference_matrices);

		tcnn::linear_kernel(assemble_rgbd<T>, 0, stream,
			output.n_elements(), 
			m_inference_density_network_output.m(), 
			m_inference_rgb_network_output.m(), 
			m_inference_density_network_output.data(), 
			m_inference_rgb_network_output.data(), 
			output.data()
		);
	}

	void density(
		cudaStream_t stream, 
		const tcnn::PitchedPtr<const float>& input, 
		tcnn::GPUMatrixDynamic<T>& output, 
		bool use_inference_matrices = true)
	{
		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = output.n();
		if (m_inference_density_network_input.n() != batch_size) {
			allocate_inference_buffers(batch_size);
		}

		m_pos_encoding->encode(
			stream,
			batch_size,
			input,
			{m_inference_density_network_input.data(), m_inference_density_network_input.m()},
			nullptr,
			use_inference_matrices
		);
		m_density_network->inference_mixed_precision(stream, m_inference_density_network_input, output, use_inference_matrices);
	}

	void density(
		cudaStream_t stream, 
		const tcnn::GPUMatrix<float>& input, 
		tcnn::GPUMatrixDynamic<T>& output, 
		bool use_inference_matrices = true) {
		density(stream, {input.data(), input.m()}, output, use_inference_matrices);
	}

	uint32_t padded_density_output_width() const {
		return m_density_network->padded_output_width();
	}

	uint32_t num_encoded_dims() const {
		return m_pos_encoding->num_encoded_dims();
	}

	void forward(
		cudaStream_t stream, 
		const tcnn::GPUMatrix<float>& input, 
		tcnn::GPUMatrixDynamic<T>* output = nullptr, 
		bool use_inference_matrices = false, 
		bool prepare_input_gradients = false
	) override 
	{
		if (output && output->layout() != tcnn::CM) {
			throw std::runtime_error("NGPNetwork::forward output must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_forward_density_network_input.n() != batch_size) {
			allocate_forward_buffers(batch_size);
		}

		// Perform positional encoding and directional encoding in parallel
		// Then perform density network query
		{
			tcnn::SyncedMultiStream synced_streams{stream, 2};

			m_pos_encoding->encode(
				synced_streams.get(0),
				batch_size,
				{input.data(), input.m()},
				{m_forward_density_network_input.data(), m_forward_density_network_input.m()},
				prepare_input_gradients ? m_forward_pos_encoding_forward_gradient.data() : nullptr,
				use_inference_matrices
			);
			m_dir_encoding->encode(
				synced_streams.get(1),
				batch_size,
				{input.data() + m_dir_offset, input.m()},
				{m_forward_rgb_network_input.data() + m_forward_density_network_output.m(), m_forward_rgb_network_input.m()},
				prepare_input_gradients ? m_forward_dir_encoding_forward_gradient.data() : nullptr,
				use_inference_matrices
			);

			m_density_network->forward(
				synced_streams.get(0), 
				m_forward_density_network_input, 
				&m_forward_density_network_output, 
				use_inference_matrices, 
				prepare_input_gradients);

			tcnn::linear_kernel(grab_density_network_output<T>, 0, synced_streams.get(0),
				m_forward_density_network_output.n_elements(), 
				m_forward_density_network_output.m(), 
				m_forward_rgb_network_input.m() /* stride */, 
				m_forward_density_network_output.data(), 
				m_forward_rgb_network_input.data()
			);
		}

		m_rgb_network->forward(
			stream, 
			m_forward_rgb_network_input, 
			&m_forward_rgb_network_output, 
			use_inference_matrices, 
			prepare_input_gradients);

		if (output) {
			tcnn::linear_kernel(assemble_rgbd<T>, 0, stream,
				output->n_elements(), 
				m_forward_density_network_output.m(), 
				m_forward_rgb_network_output.m(), 
				m_forward_density_network_output.data(), 
				m_forward_rgb_network_output.data(), 
				output->data()
			);
		}
	}

	void backward(
		cudaStream_t stream,
		const tcnn::GPUMatrix<float>& input,
		const tcnn::GPUMatrixDynamic<T>& output,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrix<float>* dL_dinput = nullptr,
		bool use_inference_matrices = false,
		bool compute_param_gradients = true
	) override 
	{
		if (output.layout() != tcnn::CM || dL_doutput.layout() != tcnn::CM) {
			throw std::runtime_error("NGPNetwork::backward output must be in column major format.");
		}

		// Make sure our temporary buffers have the correct size for the given batch size
		uint32_t batch_size = input.n();
		if (m_backward_dL_ddensity_network_output.n() != batch_size) {
			allocate_backward_buffers(batch_size);
		}

		tcnn::linear_kernel(extract_rgb<T>, 0, stream,
			batch_size*3, 
			m_backward_dL_drgb.m(), 
			dL_doutput.data(), 
			m_backward_dL_drgb.data()
		);

		m_rgb_network->backward(
			stream, 
			m_forward_rgb_network_input, 
			m_forward_rgb_network_output, 
			m_backward_dL_drgb, 
			&m_backward_dL_drgb_network_input, 
			use_inference_matrices, 
			compute_param_gradients);

		// Backprop through dir encoding if it is trainable or if we need input gradients
		tcnn::SyncedMultiStream synced_streams{stream, 2};

		if (m_dir_encoding->n_params() > 0 || dL_dinput) {
			tcnn::linear_kernel(extract_dir_gradient<T>, 0, synced_streams.get(1),
				m_backward_dL_ddir_encoding_output.n_elements(), 
				m_backward_dL_ddensity_network_output.m(), 
				m_backward_dL_ddir_encoding_output.m(), 
				m_backward_dL_drgb_network_input.m(), 
				m_backward_dL_drgb_network_input.data(), 
				m_backward_dL_ddir_encoding_output.data()
			);

			m_dir_encoding->backward(
				synced_streams.get(1),
				batch_size,
				{m_backward_dL_ddir_encoding_output.data(), m_backward_dL_ddir_encoding_output.m()},
				dL_dinput ? m_forward_dir_encoding_forward_gradient.data() : nullptr,
				dL_dinput ? tcnn::PitchedPtr<float>{dL_dinput->data() + m_dir_offset, dL_dinput->m()} : tcnn::PitchedPtr<float>{},
				{input.data() + m_dir_offset, input.m()}
			);
		}

		tcnn::linear_kernel(extract_density_gradient<T>, 0, synced_streams.get(0),
			m_backward_dL_ddensity_network_output.n_elements(),
			m_backward_dL_ddensity_network_output.m(),
			m_backward_dL_drgb_network_input.m(),
			dL_doutput.data(),
			m_forward_density_network_output.data(),
			m_backward_dL_drgb_network_input.data(),
			m_backward_dL_ddensity_network_output.data()
		);

		tcnn::GPUMatrix<T>* dL_ddensity_network_input = nullptr;
		if (m_pos_encoding->n_params() > 0 || dL_dinput) {
			dL_ddensity_network_input = &m_backward_dL_ddensity_network_input;
		}

		m_density_network->backward(
			synced_streams.get(0), 
			m_forward_density_network_input, 
			m_forward_density_network_output, 
			m_backward_dL_ddensity_network_output, 
			dL_ddensity_network_input, 
			use_inference_matrices, 
			compute_param_gradients);

		// Backprop through pos encoding if it is trainable or if we need input gradients
		if (dL_ddensity_network_input) {
			m_pos_encoding->backward(
				synced_streams.get(0),
				batch_size,
				{dL_ddensity_network_input->data(), dL_ddensity_network_input->m()},
				dL_dinput ? m_forward_pos_encoding_forward_gradient.data() : nullptr,
				dL_dinput ? tcnn::PitchedPtr<float>{dL_dinput->data(), dL_dinput->m()} : tcnn::PitchedPtr<float>{},
				{input.data(), input.m()}
			);
		}
	}

	void initialize_params(tcnn::pcg32& rnd, float* params_full_precision, T* params, T* inference_params, T* backward_params, T* gradients, float scale = 1) override {
		size_t offset = 0;
		m_density_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_density_network->n_params();

		m_rgb_network->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_rgb_network->n_params();

		m_pos_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_pos_encoding->n_params();

		m_dir_encoding->initialize_params(
			rnd,
			params_full_precision + offset,
			params + offset,
			inference_params + offset,
			backward_params + offset,
			gradients + offset,
			scale
		);
		offset += m_dir_encoding->n_params();
	}

	size_t n_params() const override {
		return m_pos_encoding->n_params() + m_density_network->n_params() + m_dir_encoding->n_params() + m_rgb_network->n_params();
	}

	uint32_t padded_output_width() const override {
		return 4;
	}

	uint32_t output_width() const override {
		return 4;
	}

	uint32_t required_input_alignment() const override {
		return 1; // No alignment required due to encoding
	}

	std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override {
		auto layers = m_density_network->layer_sizes();
		auto rgb_layers = m_rgb_network->layer_sizes();
		layers.insert(layers.end(), rgb_layers.begin(), rgb_layers.end());
		return layers;
	}

	uint32_t width(uint32_t layer) const override {
		if (layer == 0) {
			return m_forward_density_network_input.m();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->width(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_forward_rgb_network_input.m();
		} else {
			return m_rgb_network->width(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	uint32_t num_forward_activations() const override {
		return m_density_network->num_forward_activations() + m_rgb_network->num_forward_activations() + 2;
	}

	const T* forward_activations(uint32_t layer) const override {
		if (layer == 0) {
			return m_forward_density_network_input.data();
		} else if (layer < m_density_network->num_forward_activations() + 1) {
			return m_density_network->forward_activations(layer - 1);
		} else if (layer == m_density_network->num_forward_activations() + 1) {
			return m_forward_rgb_network_input.data();
		} else {
			return m_rgb_network->forward_activations(layer - 2 - m_density_network->num_forward_activations());
		}
	}

	const std::shared_ptr<tcnn::Encoding<T>>& encoding() const {
		return m_pos_encoding;
	}

	const std::shared_ptr<tcnn::Encoding<T>>& dir_encoding() const {
		return m_dir_encoding;
	}

	void auto_determine_grid_parameters(json& config) {
		// Automatically determine certain parameters if we're dealing with the (hash)grid encoding
		if (tcnn::to_lower(config.value("otype", "OneBlob")).find("grid") != std::string::npos) {
			config["n_pos_dims"] = m_n_pos_dims;

			const uint32_t n_features_per_level = config.value("n_features_per_level", 2u);

			uint32_t num_levels = config.value("n_levels", 16u);
			if (config.contains("n_features") && config["n_features"] > 0) {
				num_levels = (uint32_t)config["n_features"] / n_features_per_level;
			}

			const uint32_t log2_hashmap_size = config.value("log2_hashmap_size", 15);

			uint32_t base_grid_resolution = config.value("base_resolution", 0);
			if (!base_grid_resolution) {
				base_grid_resolution = 1u << (log2_hashmap_size / m_n_pos_dims);
				config["base_resolution"] = base_grid_resolution;
			}

			// Desired resolution of the finest hashgrid level over the unit cube
			const float desired_resolution = config.value("desired_resolution", 2048.0f); 

			// Automatically determine suitable per_level_scale
			float per_level_scale = config.value("per_level_scale", 0.0f);
			if (per_level_scale <= 0.0f && num_levels > 1) {
				per_level_scale = std::exp(std::log(desired_resolution / (float)base_grid_resolution) / (num_levels-1));
				// TODO
				// std::log(desired_resolution * (float)m_nerf.training.dataset.aabb_scale / (float)m_base_grid_resolution) / (m_num_levels-1));
				config["per_level_scale"] = per_level_scale;
			}

			tlog::info()
				<< "GridEncoding: "
				<< " Nmin=" << base_grid_resolution
				<< " b=" << per_level_scale
				<< " F=" << n_features_per_level
				<< " T=2^" << log2_hashmap_size
				<< " L=" << num_levels;
		}
	}

private:
	void allocate_inference_buffers(uint32_t batch_size) {
		m_inference_density_network_input.set_size(m_pos_encoding->num_encoded_dims(), batch_size);
		m_inference_density_network_output.set_size(m_density_network->padded_output_width(), batch_size);

		m_inference_rgb_network_input.set_size(m_rgb_network_input_width, batch_size);
		m_inference_rgb_network_output.set_size(m_rgb_network->padded_output_width(), batch_size);

		m_inference_network_output.set_size(4, batch_size);

		tcnn::GPUMatrixBase::allocate_shared_memory(
			m_inference_buffer,
			{
				&m_inference_density_network_input,
				&m_inference_density_network_output,

				&m_inference_rgb_network_input,
				&m_inference_rgb_network_output,

				&m_inference_network_output,
			}
		);

		CUDA_CHECK_THROW(cudaMemset(m_inference_rgb_network_input.data(), 0, m_inference_rgb_network_input.n_bytes()));
	}

	void allocate_forward_buffers(uint32_t batch_size) {
		m_forward_density_network_input.set_size(m_pos_encoding->num_encoded_dims(), batch_size);
		m_forward_density_network_output.set_size(m_density_network->padded_output_width(), batch_size);

		m_forward_rgb_network_input.set_size(m_rgb_network_input_width, batch_size);
		m_forward_rgb_network_output.set_size(m_rgb_network->padded_output_width(), batch_size);

		m_forward_network_output.set_size(4, batch_size);

		m_forward_pos_encoding_forward_gradient.set_size(m_pos_encoding->num_forward_gradient_dims(), batch_size);
		m_forward_dir_encoding_forward_gradient.set_size(m_dir_encoding->num_forward_gradient_dims(), batch_size);

		tcnn::GPUMatrixBase::allocate_shared_memory(
			m_forward_buffer,
			{
				&m_forward_density_network_input,
				&m_forward_density_network_output,

				&m_forward_rgb_network_input,
				&m_forward_rgb_network_output,

				&m_forward_network_output,

				&m_forward_pos_encoding_forward_gradient,
				&m_forward_dir_encoding_forward_gradient,
			}
		);

		CUDA_CHECK_THROW(cudaMemset(m_forward_rgb_network_input.data(), 0, m_forward_rgb_network_input.n_bytes()));
	}

	void allocate_backward_buffers(uint32_t batch_size) {
		m_backward_dL_drgb.set_size(m_rgb_network->padded_output_width(), batch_size);

		m_backward_dL_drgb_network_input.set_size(m_rgb_network_input_width, batch_size);
		m_backward_dL_ddensity_network_output.set_size(m_density_network->padded_output_width(), batch_size);
		m_backward_dL_ddir_encoding_output.set_size(m_dir_encoding->num_encoded_dims(), batch_size);

		m_backward_dL_ddensity_network_input.set_size(m_pos_encoding->num_encoded_dims(), batch_size);

		m_backward_dL_dpos.set_size(m_n_pos_dims, batch_size);
		m_backward_dL_ddir.set_size(m_n_dir_dims, batch_size);

		tcnn::GPUMatrixBase::allocate_shared_memory(
			m_backward_buffer,
			{
				&m_backward_dL_drgb,

				&m_backward_dL_drgb_network_input,
				&m_backward_dL_ddensity_network_output,
				&m_backward_dL_ddir_encoding_output,

				&m_backward_dL_ddensity_network_input,

				&m_backward_dL_dpos,
				&m_backward_dL_ddir,
			}
		);

		CUDA_CHECK_THROW(cudaMemset(m_backward_dL_drgb.data(), 0, m_backward_dL_drgb.n_bytes()));
	}

	uint32_t get_encoding_alignment(const json& network) {
		uint32_t alignment = 8u;
		if (network.contains("otype")) {
			if (tcnn::equals_case_insensitive(network["otype"], "FullyFusedMLP")) {
				alignment = 16u;
			} 
			else if (tcnn::equals_case_insensitive(network["otype"], "MegakernelMLP")) {
				alignment = 16u;
			}
		}
		return alignment;
	}

	std::unique_ptr<tcnn::Network<T>> 	m_density_network;
	std::unique_ptr<tcnn::Network<T>> 	m_rgb_network;
	std::shared_ptr<tcnn::Encoding<T>> 	m_pos_encoding;
	std::shared_ptr<tcnn::Encoding<T>> 	m_dir_encoding;

	uint32_t m_rgb_network_input_width;
	uint32_t m_n_pos_dims = 3;
	uint32_t m_n_dir_dims = 3;
	uint32_t m_dir_offset = 4;

	// Temporary buffers to hold inference data
	tcnn::GPUMemory<char> m_inference_buffer;
	tcnn::GPUMatrix<T> m_inference_density_network_input;
	tcnn::GPUMatrix<T> m_inference_density_network_output;
	tcnn::GPUMatrix<T> m_inference_rgb_network_input;
	tcnn::GPUMatrix<T> m_inference_rgb_network_output;
	tcnn::GPUMatrix<T> m_inference_network_output;

	// Temporary buffers to hold forward data
	tcnn::GPUMemory<char> m_forward_buffer;
	tcnn::GPUMatrix<T> m_forward_density_network_input;
	tcnn::GPUMatrix<T> m_forward_density_network_output;
	tcnn::GPUMatrix<T> m_forward_rgb_network_input;
	tcnn::GPUMatrix<T> m_forward_rgb_network_output;
	tcnn::GPUMatrix<T> m_forward_network_output; // Only needed when visualizing
	tcnn::GPUMatrix<float> m_forward_pos_encoding_forward_gradient; // Only needed when computing input gradients
	tcnn::GPUMatrix<float> m_forward_dir_encoding_forward_gradient; // Only needed when computing input gradients

	// Temporary buffers to hold backward data
	tcnn::GPUMemory<char> m_backward_buffer;
	tcnn::GPUMatrix<T> m_backward_dL_drgb;
	tcnn::GPUMatrix<T> m_backward_dL_drgb_network_input;
	tcnn::GPUMatrix<T> m_backward_dL_ddensity_network_output;
	tcnn::GPUMatrix<T> m_backward_dL_ddir_encoding_output; // Only needed when training the dir encoding or computing input gradients
	tcnn::GPUMatrix<T> m_backward_dL_ddensity_network_input;
	tcnn::GPUMatrix<float> m_backward_dL_dpos; // Only needed when computing input gradients
	tcnn::GPUMatrix<float> m_backward_dL_ddir; // Only needed when computing input gradients
};
