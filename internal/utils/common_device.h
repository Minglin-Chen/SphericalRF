#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <unsupported/Eigen/MatrixFunctions>

#include "internal/utils/common.h"
#include "internal/utils/random_val.cuh"

#include "internal/sampler/bounding_box.h"


/* *********************************************************************
 * Common
 * *********************************************************************/
inline HOST_DEVICE float4 to_float4(const Eigen::Array4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline HOST_DEVICE float4 to_float4(const Eigen::Vector4f& x) {
	return {x.x(), x.y(), x.z(), x.w()};
}

inline HOST_DEVICE float3 to_float3(const Eigen::Array3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline HOST_DEVICE float3 to_float3(const Eigen::Vector3f& x) {
	return {x.x(), x.y(), x.z()};
}

inline HOST_DEVICE float2 to_float2(const Eigen::Array2f& x) {
	return {x.x(), x.y()};
}

inline HOST_DEVICE float2 to_float2(const Eigen::Vector2f& x) {
	return {x.x(), x.y()};
}

inline HOST_DEVICE Eigen::Array4f to_array4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline HOST_DEVICE Eigen::Vector4f to_vec4(const float4& x) {
	return {x.x, x.y, x.z, x.w};
}

inline HOST_DEVICE Eigen::Array3f to_array3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline HOST_DEVICE Eigen::Vector3f to_vec3(const float3& x) {
	return {x.x, x.y, x.z};
}

inline HOST_DEVICE Eigen::Array2f to_array2(const float2& x) {
	return {x.x, x.y};
}

inline HOST_DEVICE Eigen::Vector2f to_vec2(const float2& x) {
	return {x.x, x.y};
}

/* *********************************************************************
 * Color space (i.e., sRGB or Linear)
 * *********************************************************************/
inline HOST_DEVICE float srgb_to_linear(float srgb) {
	return srgb <= 0.04045f ? srgb / 12.92f : std::pow((srgb + 0.055f) / 1.055f, 2.4f);
}

inline HOST_DEVICE Eigen::Array3f srgb_to_linear(const Eigen::Array3f& x) {
	return {srgb_to_linear(x.x()), srgb_to_linear(x.y()), (srgb_to_linear(x.z()))};
}

inline HOST_DEVICE float srgb_to_linear_derivative(float srgb) {
	return srgb <= 0.04045f ? 1.0f / 12.92f : 2.4f / 1.055f * std::pow((srgb + 0.055f) / 1.055f, 1.4f);
}

inline HOST_DEVICE Eigen::Array3f srgb_to_linear_derivative(const Eigen::Array3f& x) {
	return {srgb_to_linear_derivative(x.x()), srgb_to_linear_derivative(x.y()), (srgb_to_linear_derivative(x.z()))};
}

inline HOST_DEVICE float linear_to_srgb(float linear) {
	return linear < 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 0.41666f) - 0.055f;
}

inline HOST_DEVICE Eigen::Array3f linear_to_srgb(const Eigen::Array3f& x) {
	return {linear_to_srgb(x.x()), linear_to_srgb(x.y()), (linear_to_srgb(x.z()))};
}

inline HOST_DEVICE float linear_to_srgb_derivative(float linear) {
	return linear < 0.0031308f ? 12.92f : 1.055f * 0.41666f * std::pow(linear, 0.41666f - 1.0f);
}

inline HOST_DEVICE Eigen::Array3f linear_to_srgb_derivative(const Eigen::Array3f& x) {
	return {linear_to_srgb_derivative(x.x()), linear_to_srgb_derivative(x.y()), (linear_to_srgb_derivative(x.z()))};
}

/* *********************************************************************
 * Camera
 * *********************************************************************/
inline HOST_DEVICE float fov_to_focal_length(int resolution, float rad) {
	return 0.5f * (float)resolution / tanf(0.5f * rad);
}

inline HOST_DEVICE Eigen::Vector2f fov_to_focal_length(const Eigen::Vector2i& resolution, const Eigen::Vector2f& rads) {
	return 0.5f * resolution.cast<float>().cwiseQuotient((0.5f * rads).array().tan().matrix());
}

inline HOST_DEVICE float focal_length_to_fov(int resolution, float focal_length) {
	// return 2.f * 180.f / PI() * atanf(float(resolution)/(focal_length*2.f));
	return 180.f * M_2_PI * atanf(float(resolution)/(focal_length*2.f));
}

inline HOST_DEVICE Eigen::Vector2f focal_length_to_fov(const Eigen::Vector2i& resolution, const Eigen::Vector2f& focal_length) {
	// return 2.f * 180.f / PI() * resolution.cast<float>().cwiseQuotient(focal_length*2).array().atan().matrix();
	return 180.f * M_2_PI * resolution.cast<float>().cwiseQuotient(focal_length*2).array().atan().matrix();
}

inline HOST_DEVICE Eigen::Matrix<float, 3, 4> spec_opengl_to_opencv(
	const Eigen::Matrix<float, 3, 4> ogl_matrix, 
	const float scale, 
	const Eigen::Vector3f offset
) {
	Eigen::Matrix<float, 3, 4> ocv_matrix = ogl_matrix;
	
	ocv_matrix.col(1) *= -1.0f;
	ocv_matrix.col(2) *= -1.0f;
	ocv_matrix.col(3) = ocv_matrix.col(3) * scale + offset;

	// cycle axes: xyz <- yzx
	// Vector4f tmp = ocv_matrix.row(0);
	// ocv_matrix.row(0) = (Vector4f)ocv_matrix.row(1);
	// ocv_matrix.row(1) = (Vector4f)ocv_matrix.row(2);
	// ocv_matrix.row(2) = tmp;

	return ocv_matrix;
}

inline HOST_DEVICE Ray pixel_to_ray(
	const Eigen::Vector2i& 				pixel,
	const Eigen::Vector2i& 				resolution,
	const Eigen::Vector2f& 				focal_length,
	const Eigen::Vector2f& 				principal_point,
	const Eigen::Matrix<float, 3, 4>& 	camera_matrix,
	const uint32_t 						spp,
	const bool 							snap_to_pixel_centers
) {
	const Eigen::Vector2f offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : spp, pixel.x(), pixel.y());
	auto xy = (pixel.cast<float>() + offset).cwiseQuotient(resolution.cast<float>());

	Eigen::Vector3f origin = camera_matrix.col(3);
	Eigen::Vector3f dir = {
		(xy.x() - principal_point.x()) * (float)resolution.x() / focal_length.x(),
		(xy.y() - principal_point.y()) * (float)resolution.y() / focal_length.y(),
		1.0f
	};
	dir = camera_matrix.block<3, 3>(0, 0) * dir;
	dir = dir.normalized();

	return {origin, dir};
}

/* *********************************************************************
 * Image IO
 * *********************************************************************/
template <typename T>
__global__ void from_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4];
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	tcnn::vector_t<T, 4> rgba_out;
	float alpha = rgba[3] * (1.0f/255.0f);
	rgba_out[0] = (T)(srgb_to_linear(rgba[0] * (1.0f/255.0f)) * alpha);
	rgba_out[1] = (T)(srgb_to_linear(rgba[1] * (1.0f/255.0f)) * alpha);
	rgba_out[2] = (T)(srgb_to_linear(rgba[2] * (1.0f/255.0f)) * alpha);
	rgba_out[3] = (T)alpha;

	*((tcnn::vector_t<T, 4>*)&out[i*4]) = rgba_out;
}

/* *********************************************************************
 * Image manipulation
 * *********************************************************************/
inline __device__ Eigen::Vector2i image_pos(const Eigen::Vector2f& pos, const Eigen::Vector2i& resolution) {
	return pos.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMin(resolution - Eigen::Vector2i::Constant(1)).cwiseMax(0);
}

inline __device__ uint64_t pixel_idx(const Eigen::Vector2i& pos, const Eigen::Vector2i& resolution, uint32_t img) {
	return pos.x() + pos.y() * resolution.x() + img * (uint64_t)resolution.x() * resolution.y();
}

inline __device__ uint64_t pixel_idx(const Eigen::Vector2f& xy, const Eigen::Vector2i& resolution, uint32_t img) {
	return pixel_idx(image_pos(xy, resolution), resolution, img);
}

inline __device__ Eigen::Array4f read_rgba(Eigen::Vector2f pos, const Eigen::Vector2i& resolution, uint32_t img, const __half* training_images) {
	Eigen::Vector2i idx = image_pos(pos, resolution);

	auto read_val = [&](const Eigen::Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Eigen::Array4f{val[0], val[1], val[2], val[3]};
	};

	return read_val(idx);
}

template <uint32_t N_DIMS, typename T>
HOST_DEVICE Eigen::Matrix<float, N_DIMS, 1> read_image(const T* __restrict__ data, const Eigen::Vector2i& resolution, const Eigen::Vector2f& pos) {
	auto pos_float = Eigen::Vector2f{pos.x() * (float)(resolution.x()-1), pos.y() * (float)(resolution.y()-1)};
	Eigen::Vector2i texel = pos_float.cast<int>();

	auto weight = pos_float - texel.cast<float>();

	auto read_val = [&](Eigen::Vector2i pos) {
		pos.x() = std::max(std::min(pos.x(), resolution.x()-1), 0);
		pos.y() = std::max(std::min(pos.y(), resolution.y()-1), 0);

		Eigen::Matrix<float, N_DIMS, 1> result;
		if (std::is_same<T, float>::value) {
			result = *(Eigen::Matrix<T, N_DIMS, 1>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];
		} else {
			auto val = *(tcnn::vector_t<T, N_DIMS>*)&data[(pos.x() + pos.y() * resolution.x()) * N_DIMS];

			PRAGMA_UNROLL
			for (uint32_t i = 0; i < N_DIMS; ++i) {
				result[i] = (float)val[i];
			}
		}
		return result;
	};

	auto result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({texel.x(), texel.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({texel.x()+1, texel.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({texel.x(), texel.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({texel.x()+1, texel.y()+1})
	);

	return result;
}

inline __device__ Eigen::Array3f composit(
	Eigen::Vector2f pos, 
	const Eigen::Vector2i& resolution, 
	uint32_t img, 
	const __half* training_images, 
	const Eigen::Array3f& background_color, 
	const Eigen::Array3f& exposure_scale = Eigen::Array3f::Ones()
) {
	Eigen::Vector2i idx = image_pos(pos, resolution);

	auto read_val = [&](const Eigen::Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Eigen::Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
	};

	return read_val(idx);
}

inline __device__ Eigen::Array3f composit_and_lerp(
	Eigen::Vector2f pos, 
	const Eigen::Vector2i& resolution, 
	uint32_t img, 
	const __half* training_images, 
	const Eigen::Array3f& background_color, 
	const Eigen::Array3f& exposure_scale = Eigen::Array3f::Ones()
) {
	pos = (pos.cwiseProduct(resolution.cast<float>()) - Eigen::Vector2f::Constant(0.5f)).cwiseMax(0.0f).cwiseMin(resolution.cast<float>() - Eigen::Vector2f::Constant(1.0f + 1e-4f));

	Eigen::Vector2i pos_int = pos.cast<int>();
	auto weight = pos - pos_int.cast<float>();

	Eigen::Vector2i idx = pos_int.cwiseMin(resolution - Eigen::Vector2i::Constant(2)).cwiseMax(0);

	auto read_val = [&](const Eigen::Vector2i& p) {
		__half val[4];
		*(uint64_t*)&val[0] = ((uint64_t*)training_images)[pixel_idx(p, resolution, img)];
		return Eigen::Array3f{val[0], val[1], val[2]} * exposure_scale + background_color * (1.0f - (float)val[3]);
	};

	Eigen::Array3f result = (
		(1 - weight.x()) * (1 - weight.y()) * read_val({idx.x(), idx.y()}) +
		(weight.x()) * (1 - weight.y()) * read_val({idx.x()+1, idx.y()}) +
		(1 - weight.x()) * (weight.y()) * read_val({idx.x(), idx.y()+1}) +
		(weight.x()) * (weight.y()) * read_val({idx.x()+1, idx.y()+1})
	);

	return result;
}

template <typename T>
__global__ void discard_saturated_pixels(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, T* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4];
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	if (rgba[0]==0 && rgba[1]==0 && rgba[2]==0) {
		out[i] = (T)(0.0f);
	}

	if (rgba[0]==255 && rgba[1]==255 && rgba[2]==255) {
		out[i] = (T)(0.0f);
	}
}

/* *********************************************************************
 * Network input & output
 * *********************************************************************/
inline __device__ float network_to_rgb(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return val;
		case EActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case EActivation::Logistic: return tcnn::logistic(val);
		case EActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_rgb_derivative(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return 1.0f;
		case EActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case EActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case EActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_density(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return val;
		case EActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case EActivation::Logistic: return tcnn::logistic(val);
		case EActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ float network_to_density_derivative(float val, EActivation activation) {
	switch (activation) {
		case EActivation::None: return 1.0f;
		case EActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case EActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case EActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

inline __device__ Eigen::Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, EActivation activation) {
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}

/* *********************************************************************
 * Coordinate transformation
 * *********************************************************************/
inline __device__ Eigen::Vector3f warp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb, EWarpingType type) 
{
	Eigen::Vector3f z;

	switch (type) {
	case EWarpingType::NGP :
		{
			z = aabb.relative_pos(pos);
		}
		break;
	case EWarpingType::MipNeRF360 :
		{
			Eigen::Vector3f x = (pos - Eigen::Vector3f{0.5f, 0.5f, 0.5f}) * 2.f;
			// MipNeRF 360 contraction
			float x_mag = x.norm();
			float x_mag_inv = 1.f / x_mag;
			z = (x_mag <= 1.f) ? x : ((2.f - x_mag_inv) * x_mag_inv * x);
			// transform from [-2,2] to [0,1]
			z = z * 0.25f + Eigen::Vector3f{0.5f, 0.5f, 0.5f};
		}
		break;
	case EWarpingType::DONeRF :
		{
			float dmax = aabb.diag().norm();
			Eigen::Vector3f x = pos - Eigen::Vector3f{0.5f, 0.5f, 0.5f};
			// DONeRF radial distortion using inverse square root transform
			float x_mag = x.norm();
			z = (x_mag == 0.f) ? x : (x / sqrt(x_mag * dmax));
			// transform from [-1,1] to [0,1]
			z = z * 0.5f + Eigen::Vector3f{0.5f, 0.5f, 0.5f};
		}
		break;
	case EWarpingType::Sigmoid :
		{
			z = {tcnn::logistic(pos.x() - 0.5f), tcnn::logistic(pos.y() - 0.5f), tcnn::logistic(pos.z() - 0.5f)};
		}
		break;
	case EWarpingType::ERP:
		{
			Eigen::Vector3f p = pos - Eigen::Vector3f{0.5f, 0.5f, 0.5f};
			float p_mag = p.norm();
			float p_mag_inv = 1.f / p_mag;
			// (-pi, pi]
			float theta = atan2f(p.y(), p.x());
			// (-pi/2, pi/2]
			float phi	= asinf(p.z() * p_mag_inv);
			// [0.0, 1.0]
			p_mag *= 2.f;
			p_mag = (p_mag <= 1.f) ? p_mag : (2.f - p_mag_inv);
			p_mag *= 0.5f;
			
			z = Eigen::Vector3f{
				(theta * M_1_PI + 1.0f) * 0.5f,		// (0.0, 1.0]
				phi * M_1_PI + 0.5f, 				// (0.0, 1.0]
				p_mag								// [0.0, 1.0]
			};
		}
	case EWarpingType::SinusoidalProjection:
		{
			Eigen::Vector3f p = pos - Eigen::Vector3f{0.5f, 0.5f, 0.5f};
			float p_mag = p.norm();
			float p_mag_inv = 1.f / p_mag;
			float theta = atan2f(p.y(), p.x());								// (-pi, pi]
			float phi 	= asinf(p.z() * p_mag_inv);							// [-pi/2, pi/2]
			p_mag = (p_mag <= 1.f) ? p_mag : (2.f - p_mag_inv);
			p_mag *= 0.5f;													// [0.0, 1.0]

			z = Eigen::Vector3f{
					theta*cosf(phi)*M_1_PI*0.5f + 0.5f, // [0.0, 1.0]
					phi*M_1_PI + 0.5f, 					// [0.0, 1.0]
					p_mag								// [0.0, 1.0]
				};
		}
	default:
		break;
	}

	return z;
}

inline __device__ Eigen::Vector3f unwarp_position(const Eigen::Vector3f& pos, const BoundingBox& aabb, EWarpingType type) 
{
	Eigen::Vector3f x;

	switch (type) {
	case EWarpingType::NGP : 
		{
			x = aabb.min + pos.cwiseProduct(aabb.diag());
		}
		break;
	case EWarpingType::MipNeRF360 :
		{
			// transform from [0,1] to [-2,2]
			Eigen::Vector3f z = (pos - Eigen::Vector3f{0.5f, 0.5f, 0.5f}) * 4.f;
			// MipNeRF 360 inverse contraction
			float z_mag = z.norm();
			float z_mag_inv = 1.f / z_mag;
			x = (z_mag <= 1.f) ? z : (1.f / (2.f - z_mag) * z_mag_inv * z);

			x = x * 0.5f + Eigen::Vector3f{0.5f, 0.5f, 0.5f};
		}
		break;
	case EWarpingType::DONeRF :
		{
			float dmax = aabb.diag().norm();
			// transform from [0,1] to [-1,1]
			Eigen::Vector3f z = (pos - Eigen::Vector3f{0.5f, 0.5f, 0.5f}) * 2.f;
			// DONeRF inverse radial distortion
			float z_mag = z.norm();
			x = z * z_mag * dmax;

			x = x + Eigen::Vector3f{0.5f, 0.5f, 0.5f};
		}
		break;
	case EWarpingType::Sigmoid :
		{
			x = Eigen::Vector3f{tcnn::logit(pos.x()) + 0.5f, tcnn::logit(pos.y()) + 0.5f, tcnn::logit(pos.z()) + 0.5f};
		}
		break;
	case EWarpingType::ERP:
		{
			float theta = (pos.x() * 2.f - 1.f) * M_PI;
			float phi	= (pos.y() - 0.5f) * M_PI;
			float p_mag = pos.z() * 2.f;
			p_mag = (p_mag <= 1.f) ? p_mag : 1.f / (2.f - p_mag);
			p_mag *= .5f;

			x = Eigen::Vector3f{
				p_mag * sinf(phi) * cosf(theta),
				p_mag * sinf(phi) * sinf(theta),
				p_mag * cosf(phi)};
			x = x + Eigen::Vector3f{0.5f, 0.5f, 0.5f};
		}
	case EWarpingType::SinusoidalProjection:
		{
			float phi 	= (pos.y() - 0.5f) * M_PI;
			float theta = (pos.x() - 0.5f) * 2.f * M_PI / cosf(phi);
			float p_mag = pos.z() * 2.f;
			p_mag = (p_mag <= 1.f) ? p_mag : 1.f / (2.f - p_mag);

			x = Eigen::Vector3f{
				p_mag * cosf(phi) * cosf(theta), 
				p_mag * cosf(phi) * sinf(theta),
				p_mag * sinf(phi)};
			x = x + Eigen::Vector3f{0.5f, 0.5f, 0.5f};
		}
	default:
		break;
	}

	return x;
}

inline __device__ Eigen::Vector3f warp_direction(const Eigen::Vector3f& dir) {
	return (dir + Eigen::Vector3f::Ones()) * 0.5f;
}

inline __device__ Eigen::Vector3f unwarp_direction(const Eigen::Vector3f& dir) {
	return dir * 2.0f - Eigen::Vector3f::Ones();
}

inline __device__ Eigen::Vector3f warp_direction_derivative(const Eigen::Vector3f& dir) {
	return Eigen::Vector3f::Constant(0.5f);
}

inline __device__ Eigen::Vector3f unwarp_direction_derivative(const Eigen::Vector3f& dir) {
	return Eigen::Vector3f::Constant(2.0f);
}

inline __device__ float warp_dt(float dt) {
	return dt;
}

inline __device__ float unwarp_dt(float dt) {
	return dt;
}