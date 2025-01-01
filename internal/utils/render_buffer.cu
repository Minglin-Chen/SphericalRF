#include "internal/utils/render_buffer.h"
#include "internal/utils/common_device.h"

#include <cuda_gl_interop.h>

using namespace Eigen;
using namespace tcnn;


std::atomic<size_t> g_total_n_bytes_allocated{0};


void CudaSurface2D::free() {
	if (m_surface) {
		cudaDestroySurfaceObject(m_surface);
	}
	m_surface = 0;
	if (m_array) {
		cudaFreeArray(m_array);
		g_total_n_bytes_allocated -= m_size.prod() * sizeof(float4);
	}
	m_array = nullptr;
}

void CudaSurface2D::resize(const Vector2i& size) {
	if (size == m_size) {
		return;
	}

	free();

	m_size = size;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
	CUDA_CHECK_THROW(cudaMallocArray(&m_array, &desc, size.x(), size.y(), cudaArraySurfaceLoadStore));

	g_total_n_bytes_allocated += m_size.prod() * sizeof(float4);

	struct cudaResourceDesc resource_desc;
	memset(&resource_desc, 0, sizeof(resource_desc));
	resource_desc.resType = cudaResourceTypeArray;
	resource_desc.res.array.array = m_array;
	CUDA_CHECK_THROW(cudaCreateSurfaceObject(&m_surface, &resource_desc));
}

__global__ void accumulate_kernel(Vector2i resolution, Array4f* frame_buffer, Array4f* accumulate_buffer, float sample_count, EColorSpace color_space) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	Array4f color = frame_buffer[idx];
	Array4f tmp = accumulate_buffer[idx];

	switch (color_space) {
		case EColorSpace::VisPosNeg:
			{
				float val = color.x() - color.y();
				float tmp_val = tmp.x() - tmp.y();

				tmp_val = (tmp_val * sample_count + val) / (sample_count+1);

				tmp.x() = fmaxf(tmp_val, 0.0f);
				tmp.y() = fmaxf(-tmp_val, 0.0f);
				break;
			}
		case EColorSpace::SRGB:
			color.head<3>() = linear_to_srgb(color.head<3>());
			// fallthrough is intended!
		case EColorSpace::Linear:
			tmp.head<3>() = (tmp.head<3>() * sample_count + color.head<3>()) / (sample_count+1); break;
	}

	tmp.w() = (tmp.w() * sample_count + color.w()) / (sample_count+1);

	accumulate_buffer[idx] = tmp;
}

__device__ Array3f tonemap(Array3f x, ETonemapCurve curve) {
	if (curve == ETonemapCurve::Identity) {
		return x;
	}

	x = x.cwiseMax(0.f);

	float k0, k1, k2, k3, k4, k5;
	if (curve == ETonemapCurve::ACES) {
		// Source:  ACES approximation : https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
		// Include pre - exposure cancelation in constants
		k0 = 0.6f * 0.6f * 2.51f;
		k1 = 0.6f * 0.03f;
		k2 = 0.0f;
		k3 = 0.6f * 0.6f * 2.43f;
		k4 = 0.6f * 0.59f;
		k5 = 0.14f;
	} else if (curve == ETonemapCurve::Hable) {
		// Source: https://64.github.io/tonemapping/
		const float A = 0.15f;
		const float B = 0.50f;
		const float C = 0.10f;
		const float D = 0.20f;
		const float E = 0.02f;
		const float F = 0.30f;
		k0 = A * F - A * E;
		k1 = C * B * F - B * E;
		k2 = 0.0f;
		k3 = A * F;
		k4 = B * F;
		k5 = D * F * F;

		const float W = 11.2f;
		const float nom = k0 * (W*W) + k1 * W + k2;
		const float denom = k3 * (W*W) + k4 * W + k5;
		const float whiteScale = denom / nom;

		// Include white scale and exposure bias in rational polynomial coefficients
		k0 = 4.0f * k0 * whiteScale;
		k1 = 2.0f * k1 * whiteScale;
		k2 = k2 * whiteScale;
		k3 = 4.0f * k3;
		k4 = 2.0f * k4;
	} else { //if (curve == ETonemapCurve::Reinhard)
		const Vector3f luminanceCoefficients = Vector3f(0.2126f, 0.7152f, 0.0722f);
		float Y = luminanceCoefficients.dot(x.matrix());

		return x * (1.f/(Y + 1.0f));
	}

	Array3f colSq = x * x;
	Array3f nom = colSq * k0 + k1 * x + k2;
	Array3f denom = k3 * colSq + k4 * x + k5;

	Array3f toneMappedCol = nom / denom;

	return toneMappedCol;
}

__global__ void tonemap_kernel(Vector2i resolution, float exposure, Array4f background_color, Array4f* accumulate_buffer, EColorSpace color_space, EColorSpace output_color_space, ETonemapCurve tonemap_curve, cudaSurfaceObject_t surface) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	// The accumulate buffer is always linear
	Array4f color = accumulate_buffer[idx];
	float scale = powf(2.0f, exposure);
	color.head<3>() *= scale;

	// The background color is represented in SRGB, so convert
	// to linear if that's not the space in which we're rendering.
	if (color_space != EColorSpace::SRGB) {
		background_color.head<3>() = srgb_to_linear(background_color.head<3>());
	}

	float weight = (1 - color.w()) * background_color.w();
	color.head<3>() += background_color.head<3>() * weight;
	color.w() += weight;

	// Conversion to output by
	// 1. converting to linear. (VisPosNeg is treated as linear red/green)
	if (color_space == EColorSpace::SRGB) {
		color.head<3>() = srgb_to_linear(color.head<3>());
	}

	color.head<3>() = tonemap(color.head<3>(), tonemap_curve);

	// 2. converting to output color space.
	if (output_color_space == EColorSpace::SRGB) {
		color.head<3>() = linear_to_srgb(color.head<3>());
	}

	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

void CudaRenderBuffer::resize(const Vector2i& size) {
	auto prev_res = resolution();

	m_surface_provider->resize(size);

	auto res = resolution();
	m_frame_buffer.enlarge((size_t)res.x() * res.y());
	m_accumulate_buffer.enlarge((size_t)res.x() * res.y());

	if (res != prev_res) {
		reset_accumulation();
	}
}

void CudaRenderBuffer::clear_frame_buffer(cudaStream_t stream) {
	auto res = resolution();
	CUDA_CHECK_THROW(cudaMemsetAsync(frame_buffer(), 0, sizeof(Array4f) * res.x() * res.y(), stream));
}

void CudaRenderBuffer::accumulate(cudaStream_t stream) {
	auto res = resolution();

	if (m_spp == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(accumulate_buffer(), 0, sizeof(Array4f) * res.x() * res.y(), stream));
	}

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	accumulate_kernel<<<blocks, threads, 0, stream>>>(
		res,
		frame_buffer(),
		accumulate_buffer(),
		(float)m_spp,
		m_color_space
	);

	++m_spp;
}

void CudaRenderBuffer::tonemap(float exposure, const Array4f& background_color, EColorSpace output_color_space, cudaStream_t stream) {
	auto res = resolution();
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res.x(), threads.x), div_round_up((uint32_t)res.y(), threads.y), 1 };
	tonemap_kernel<<<blocks, threads, 0, stream>>>(
		res,
		exposure,
		background_color,
		accumulate_buffer(),
		m_color_space,
		output_color_space,
		m_tonemap_curve,
		surface()
	);
}