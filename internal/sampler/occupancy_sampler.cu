#include "internal/utils/common.h"
#include "internal/sampler/bounding_box.h"
#include "internal/utils/random_val.cuh"
#include "internal/testbed.h"

using namespace Eigen;
using namespace tcnn;


inline constexpr __device__ float SQRT3() { return 1.73205080757f; }

__global__ void mark_untrained_density_grid(
	const uint32_t n_elements, 
	const uint32_t n_grid_size,
	const uint32_t n_cascades,
	const uint32_t n_images,
	const Vector2i resolution,
	const Vector2f focal_length,
	const Matrix<float, 3, 4>* training_xforms,
	const uint32_t aabb_scale,
	const BoundingBox aabb,
	EWarpingType warping_type,
	EGridDistType griddist_type,
	float* __restrict__ grid_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t level 		= i / (n_grid_size*n_grid_size*n_grid_size);
	uint32_t pos_idx 	= i % (n_grid_size*n_grid_size*n_grid_size);

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	// position in metric space
	Vector3f pos = (Vector3f{(float)x+0.5f, (float)y+0.5f, (float)z+0.5f}) / n_grid_size - Vector3f::Constant(0.5f);
	switch (griddist_type)
	{
	case EGridDistType::WarpedSpace_Exp :
		pos *= scalbnf(1.0f, -(n_cascades-level-1));
		pos += Vector3f::Constant(0.5f);
		// warped space -> metric space
		pos = unwarp_position(pos, aabb, warping_type);
		break;
	case EGridDistType::WarpedSpace_Reciprocal :
		pos /= (n_cascades-level);
		pos += Vector3f::Constant(0.5f);
		// warped space -> metric space
		pos = unwarp_position(pos, aabb, warping_type);
		break;
	case EGridDistType::MetricSpace_Exp :
		pos *= scalbnf(float(aabb_scale), -(n_cascades-level-1));
		pos += Vector3f::Constant(0.5f);
	case EGridDistType::MetricSpace_Reciprocal :
		pos *= float(aabb_scale) / (n_cascades-level);
		pos += Vector3f::Constant(0.5f);
	default:
		break;
	}
	
	float half_resx = resolution.x() * 0.5f;
	float half_resy = resolution.y() * 0.5f;
	float voxel_radius = 0.5f * SQRT3() * aabb_scale / n_grid_size;
	switch (griddist_type)
	{
	case EGridDistType::WarpedSpace_Exp :
		// It is unreasonable! Recommend not use in this case!
		voxel_radius *= scalbnf(1.0f, -(n_cascades-level-1));
		break;
	case EGridDistType::WarpedSpace_Reciprocal :
		// It is unreasonable! Recommend not use in this case!
		voxel_radius /= (n_cascades-level);
		break;
	case EGridDistType::MetricSpace_Exp :
		voxel_radius *= scalbnf(1.0f, -(n_cascades-level-1));
		break;
	case EGridDistType::MetricSpace_Reciprocal :
		voxel_radius /= (n_cascades-level);
		break;
	default:
		break;
	}

	int count = 0;
	for (uint32_t j=0; j<n_images; ++j) {
		Matrix<float, 3, 4> xform = training_xforms[j];
		Vector3f ploc = pos - xform.col(3);
		float x = ploc.dot(xform.col(0));
		float y = ploc.dot(xform.col(1));
		float z = ploc.dot(xform.col(2));
		if (z > 0.f) {
			auto focal = focal_length;
			// TODO - add a box / plane intersection to stop thomas from murdering me
			if (fabsf(x)-voxel_radius < z/focal.x()*half_resx && fabsf(y)-voxel_radius < z/focal.y()*half_resy) {
				count++;
				if (count > 0) break;
			}
		}
	}
	grid_out[i] = (count > 0) ? 0.f : -1.f;
}

__global__ void generate_grid_samples_nerf_nonuniform(
	const uint32_t n_elements, 
	default_rng_t rng, 
	const uint32_t step, 
	EWarpingType warping_type, 
	const uint32_t aabb_scale,
	const BoundingBox aabb, 
	EGridDistType griddist_type,
	const float* __restrict__ grid_in, 
	Position* __restrict__ out, 
	uint32_t* __restrict__ indices, 
	const uint32_t n_grid_size, 
	const uint32_t n_grid_elements, 
	const uint32_t n_cascades, 
	const float min_cone_stepsize, 
	const float thresh 
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// 1 random number to select the level, 3 to select the position.
	rng.advance(i*4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// select grid cell that has density
	uint32_t idx;
	for (uint32_t j = 0; j < 10; ++j) {
		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % n_grid_elements;
		idx += level * n_grid_elements;
		if (grid_in[idx] > thresh) {
			break;
		}
	}

	// random position within that cell
	uint32_t pos_idx = idx % n_grid_elements;

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	// position in warped space
	Vector3f pos = (Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / n_grid_size - Vector3f::Constant(0.5f);
	switch (griddist_type)
	{
	case EGridDistType::WarpedSpace_Exp :
		// L: 0 -> N-1
		// pos * 2^(-(N-L-1)): pos*2^(-(N-1)) -> pos
		// trend: increase
		pos *= scalbnf(1.0f, -(n_cascades-level-1));
		pos += Vector3f::Constant(0.5f);
		break;
	case EGridDistType::WarpedSpace_Reciprocal :
		// L: 0 -> N-1
		// pos * 1/(N-L): pos/N -> pos
		// trend: increase
		pos *= 1.f / (n_cascades-level);
		pos += Vector3f::Constant(0.5f);
		break;
	case EGridDistType::MetricSpace_Exp :
		// L: 0 -> N-1
		// pos * 2^(-(N-L-1)) * aabb_length: pos*2^(-(N-1))*aabb_length -> pos*aabb_length
		// trend: increase
		pos *= scalbnf(float(aabb_scale), -(n_cascades-level-1));
		pos += Vector3f::Constant(0.5f);
		pos = warp_position(pos, aabb, warping_type);
		break;
	case EGridDistType::MetricSpace_Reciprocal :
		// L: 0 -> N-1
		// pos * 1/(N-L) * aabb_length: pos/N*aabb_length -> pos*aabb_length
		// trend: increase
		pos *= float(aabb_scale) / (n_cascades-level);
		pos += Vector3f::Constant(0.5f);
		pos = warp_position(pos, aabb, warping_type);
		break;
	default:
		break;
	}

	out[i] = { pos, warp_dt(min_cone_stepsize) };
	indices[i] = idx;
}

__global__ void splat_grid_samples_nerf_max_nearest_neighbor(
	const uint32_t n_elements, 
	const uint32_t* __restrict__ indices, 
	const int padded_output_width, 
	const tcnn::network_precision_t* network_output, 
	const EActivation density_activation,
	const uint32_t n_grid_size,
	const float min_cone_stepsize,
	float* __restrict__ grid_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t local_idx = indices[i];

	// Current setting: optical thickness of the smallest possible stepsize.
	// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
	uint32_t level = 0;//local_idx / (n_grid_size * n_grid_size * n_grid_size);

	float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float optical_thickness = mlp * scalbnf(min_cone_stepsize, level);

	// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
	// uint atomicMax is thus perfectly acceptable.
	atomicMax((uint32_t*)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

__global__ void ema_grid_samples_nerf(
	const uint32_t n_elements,
	const float* __restrict__ grid_in,
	float decay,
	const uint32_t count,
	float* __restrict__ grid_out
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float importance = grid_in[i];

	// float ema_debias_old = 1 - (float)powf(decay, count);
	// float ema_debias_new = 1 - (float)powf(decay, count+1);

	// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
	// grid_out[i] = filtered_val;

	// Maximum instead of EMA allows capture of very thin features.
	// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

	float prev_val = grid_out[i];
	float val = (prev_val<0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	grid_out[i] = val;
}

__global__ void grid_to_bitfield(
	const uint32_t n_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr,
	const float min_optical_thickness
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	float thresh = std::min(min_optical_thickness, *mean_density_ptr);

	#pragma unroll
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(
	const uint32_t n_elements,
	const uint32_t n_grid_size,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	#pragma unroll
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = tcnn::morton3D_invert(i>>0) + n_grid_size/8;
	uint32_t y = tcnn::morton3D_invert(i>>1) + n_grid_size/8;
	uint32_t z = tcnn::morton3D_invert(i>>2) + n_grid_size/8;

	next_level[tcnn::morton3D(x, y, z)] |= bits;
}

inline __device__ float calc_dt(
	const float t, 
	const uint32_t n_max_steps,
	const float cone_angle, 
	const float min_cone_stepsize,
	const float max_cone_stepsize,
	const float near_distance,
	const float far_distance,
	ESamplingType type
) {
	float dt = 0.f;
	float g_tfar, g_tnear, g_t0, g_t1, t1;
	float ds = 1.f / (float)(n_max_steps);

	switch (type) {
	case ESamplingType::Original : 
		{
			dt = tcnn::clamp(t*cone_angle, min_cone_stepsize, max_cone_stepsize);
		}
		break;
	case ESamplingType::Disparity :
		{
			g_tfar = 1.f / far_distance; g_tnear = 1.f / near_distance; g_t0 = 1.f / t;
			g_t1 = ds * (g_tfar - g_tnear) + g_t0;
			t1 = 1.f / g_t1;
			dt = t1 - t;
			if (dt < 0.f) {
				dt = t >= far_distance ? 0.f : far_distance - t;
			}
		}
		break;
	case ESamplingType::Log :
		{
			g_tfar = __logf(far_distance); g_tnear = __logf(near_distance); g_t0 = __logf(t);
			g_t1 = ds * (g_tfar - g_tnear) + g_t0;
			t1 = __expf(g_t1);
			t1 = t1 < far_distance ? t1 : far_distance;
			dt = t1 - t;
		}
		break;
	default:
		break;
	}
	return dt;
}

inline __device__ int mip_from_pos(const Vector3f& pos, uint32_t n_cascades) {
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(n_cascades-1, max(0, exponent+1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f& pos, uint32_t n_grid_size, uint32_t n_cascades) {
	int mip = mip_from_pos(pos, n_cascades);
	dt *= 2 * n_grid_size;
	if (dt<1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(n_cascades-1, max(exponent, mip));
}

inline __device__ int level_from_pos(
	const Vector3f& pos, 
	const Vector3f& warped_pos, 
	const BoundingBox aabb,
	uint32_t n_cascades, 
	EGridDistType griddist_type) 
{
	float maxval = 0.0;
	switch (griddist_type)
	{
	case EGridDistType::WarpedSpace_Exp :
	case EGridDistType::WarpedSpace_Reciprocal :
		maxval = (warped_pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
		break;
	case EGridDistType::MetricSpace_Exp :
	case EGridDistType::MetricSpace_Reciprocal :
		maxval = (pos - Vector3f::Constant(0.5f)).cwiseQuotient(aabb.diag()).cwiseAbs().maxCoeff();
		break;
	default:
		break;
	}
	maxval *= 2.f;

	int mip = 0;
	for(; mip<n_cascades-1; mip++) {
		bool got = false;

		switch (griddist_type)
		{
		case EGridDistType::WarpedSpace_Exp :
			got = maxval > scalbnf(1.f, -1-mip);
			break;
		case EGridDistType::WarpedSpace_Reciprocal :
			got = maxval > 1.f / (mip+2);
			break;
		case EGridDistType::MetricSpace_Exp :
			got = maxval > scalbnf(1.f, -1-mip);
			break;
		case EGridDistType::MetricSpace_Reciprocal :
			got = maxval > 1.f / (mip+2);
			break;
		default:
			break;
		}

		if (got) break;
	}

	return n_cascades-mip-1;
}

inline HOST_DEVICE uint32_t grid_mip_offset(uint32_t mip, uint32_t n_grid_size) {
	return (n_grid_size * n_grid_size * n_grid_size) * mip;
}

__device__ uint32_t cascaded_grid_idx_at(
	Vector3f pos, 
	Vector3f warped_pos,
	BoundingBox aabb,
	uint32_t level, 
	uint32_t n_grid_size,
	uint32_t n_cascades,
	EGridDistType griddist_type
) {
	Vector3f p = {0.f, 0.f, 0.f};
	switch (griddist_type)
	{
	case EGridDistType::WarpedSpace_Exp :
		p = warped_pos - Vector3f::Constant(0.5f);
		p *= scalbnf(1.0f, n_cascades-level-1);
		p += Vector3f::Constant(0.5f);
		break;
	case EGridDistType::WarpedSpace_Reciprocal :
		p = warped_pos - Vector3f::Constant(0.5f);
		p *= (n_cascades-level);
		p += Vector3f::Constant(0.5f);
		break;
	case EGridDistType::MetricSpace_Exp :
		p = pos - Vector3f::Constant(0.5f);
		p = p.cwiseQuotient(aabb.diag());
		p *= scalbnf(1.0f, n_cascades-level-1);
		p += Vector3f::Constant(0.5f);
		break;
	case EGridDistType::MetricSpace_Reciprocal :
		p = pos - Vector3f::Constant(0.5f);
		p = p.cwiseQuotient(aabb.diag());
		p *= (n_cascades-level);
		p += Vector3f::Constant(0.5f);
		break;
	default:
		break;
	}
	
	Vector3i i = (p * n_grid_size).cast<int>();

	if (i.x() < -1 || i.x() > n_grid_size || i.y() < -1 || i.y() > n_grid_size || i.z() < -1 || i.z() > n_grid_size) {
		printf("WTF %d %d %d\n", i.x(), i.y(), i.z());
	}

	uint32_t idx = tcnn::morton3D(
		tcnn::clamp(i.x(), 0, (int)n_grid_size-1),
		tcnn::clamp(i.y(), 0, (int)n_grid_size-1),
		tcnn::clamp(i.z(), 0, (int)n_grid_size-1)
	);

	return idx;
}

__device__ bool density_grid_occupied_at(
	const Vector3f& pos, 
	const Vector3f& warped_pos, 
	const BoundingBox aabb,
	const uint8_t* density_grid_bitfield, 
	uint32_t level, 
	uint32_t n_grid_size,
	uint32_t n_cascades,
	EGridDistType griddist_type
) {
	uint32_t idx = cascaded_grid_idx_at(pos, warped_pos, aabb, level, n_grid_size, n_cascades, griddist_type);
	return density_grid_bitfield[idx/8+grid_mip_offset(level,n_grid_size)/8] & (1<<(idx%8));
}

inline __device__ float distance_to_next_voxel(
	const Eigen::Vector3f& pos, 
	const Eigen::Vector3f& dir, 
	const Eigen::Vector3f& idir, 
	uint32_t res
) { // dda like step
	Eigen::Vector3f p = res * pos;
	float tx = (floorf(p.x() + 0.5f + 0.5f * copysignf(1.f, dir.x())) - p.x()) * idir.x();
	float ty = (floorf(p.y() + 0.5f + 0.5f * copysignf(1.f, dir.y())) - p.y()) * idir.y();
	float tz = (floorf(p.z() + 0.5f + 0.5f * copysignf(1.f, dir.z())) - p.z()) * idir.z();
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(
	float t,
	const uint32_t n_max_steps,
	const float cone_angle,
	const float min_cone_stepsize,
	const float max_cone_stepsize,
	const float near_distance,
	const float far_distance,
	ESamplingType sampling_type,
	const Eigen::Vector3f& pos,
	const Eigen::Vector3f& dir,
	const Eigen::Vector3f& idir,
	uint32_t res
) {
	// Analytic stepping by a multiple of dt. Make empty space unequal to non-empty space
	// due to the different stepping.
	// float dt = calc_dt(t, cone_angle, min_cone_stepsize, max_cone_stepsize);
	// return t + ceilf(fmaxf(distance_to_next_voxel(pos, dir, idir, res) / dt, 0.5f)) * dt;

	// Regular stepping (may be slower but matches non-empty space)
	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);
	do {
		t += calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
	} while (t < t_target);
	return t;
}

__global__ void generate_training_samples_nerf(
	const uint32_t n_rays,
	const uint32_t n_rays_shift,
	const uint32_t n_coords,
	// dataset
	const uint32_t n_images,
	const Vector2i resolution,
	const Vector2f focal_length,
	const Vector2f principal_point,
	const Matrix<float, 3, 4>* __restrict__ training_xforms,
	const __half* __restrict__ training_images,
	const __half* __restrict__ sampling_probability,
	// warping parameters
	EWarpingType warping_type,
	// grid
	BoundingBox aabb,
	const uint32_t n_grid_size,
	const uint32_t n_cascades,
	EGridDistType griddist_type,
	const uint8_t* __restrict__ density_grid,
	const bool use_grid,
	// sampling parameters
	const bool snap_to_pixel_centers,
	const uint32_t n_max_steps,
	const float cone_angle_constant,
	const float min_cone_stepsize,
	const float max_cone_stepsize,
	const float near_distance,
	const float far_distance,
	ESamplingType sampling_type,
	// output rays & coordinates
	uint32_t* __restrict__ ray_counter,				/* 1 */				// #sampled rays which having suitable sampled points
	uint32_t* __restrict__ numsteps_counter,		/* 1 */				// #sampled points of all rays
	Ray* __restrict__ rays_out,						/* n_rays */		// sampled rays
	Array4f* __restrict__ rays_rgba,				/* n_rays */		// ray rgba
	uint32_t* __restrict__ numsteps_out,			/* n_rays x 2 */ 	// first: #sampled points, second: start index of the sampled points
	Coordinate* __restrict__ coords_out,			/* n_coords */		// sampled points of all rays
	default_rng_t rng
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	// sample the image index
	const uint32_t img = (((n_rays_shift + i) * n_images) / n_rays) % n_images;

	Matrix<float, 3, 4> xform = training_xforms[img];

	// sample the pixel location
	// 8 is used to index into the PRNG stream. Must be larger than the number of
	// samples consumed by any given training ray.
	rng.advance(i * 8);
	Vector2f xy = random_val_2d(rng);
	if (snap_to_pixel_centers) {
		xy = (xy.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMax(0).cwiseMin(resolution - Vector2i::Ones()).cast<float>() + Vector2f::Constant(0.5f)).cwiseQuotient(resolution.cast<float>());
	}

	// sampling probability
	float s_prob = sampling_probability[pixel_idx(image_pos(xy, resolution), resolution, img)];
	if (s_prob <= 0.f) return;

	// 1. generate the ray (i.e., the original and normalized view direction of the ray)
	Ray ray;
	// - rays need to be inferred from the camera matrix
	ray.o = xform.col(3);
	ray.d = {
		(xy.x()-principal_point.x())*resolution.x() / focal_length.x(),
		(xy.y()-principal_point.y())*resolution.y() / focal_length.y(),
		1.0f,
	};
	ray.d = (xform.block<3, 3>(0, 0) * ray.d).normalized();

	// 2. sample
	Vector2f tminmax = aabb.ray_intersect(ray.o, ray.d);
	// - the near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x() = fmaxf(tminmax.x(), near_distance);

	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();
	// calc_cone_angle(ray.d.dot(xform.col(2)), focal_length, cone_angle_constant);
	float cone_angle = cone_angle_constant;

	float startt = tminmax.x();
	startt += calc_dt(startt, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type) * random_val(rng);
	Vector3f idir = ray.d.cwiseInverse();

	// - 2.1 first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	Vector3f pos;
	if (use_grid) {
		while (aabb.contains(pos = ray.o + t * ray.d) && j < n_max_steps) {
			// float dt = calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			// uint32_t mip = mip_from_dt(dt, pos, n_grid_size, n_cascades);
			// if (density_grid_occupied_at(pos, density_grid, mip, n_grid_size, n_cascades, griddist_type)) {
			// 	++j;
			// 	t += dt;
			// } else {
			// 	uint32_t res = n_grid_size>>mip;
			// 	t = advance_to_next_voxel(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type, pos, ray.d, idir, res);
			// }

			float dt = calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			Vector3f warped_pos = warp_position(pos, aabb, warping_type);
			uint32_t level = level_from_pos(pos, warped_pos, aabb, n_cascades, griddist_type);
			if (density_grid_occupied_at(pos, warped_pos, aabb, density_grid, level, n_grid_size, n_cascades, griddist_type)) {
				++j;
			}
			t += dt;
			if (dt==0) break;
		}
	} else {
		while (aabb.contains(pos = ray.o + t * ray.d) && j < n_max_steps) {
			float dt = calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			++j;
			t += dt;
			if (dt==0) break;
		}
	}
	
	if (j == 0) {
		return;
	}
	uint32_t numsteps = j;
	// - first entry in the array is a counter
	uint32_t base = atomicAdd(numsteps_counter, numsteps);
	if (base + numsteps > n_coords) {
		return;
	}

	// - 2.1 second pass to generate sampled points
	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	rays_out[ray_idx] = ray;
	rays_rgba[ray_idx] = read_rgba(xy, resolution, img, training_images);
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	Vector3f warped_dir = warp_direction(ray.d);
	j = 0;
	t = startt;
	if (use_grid) {
		while (aabb.contains(pos = ray.o + t * ray.d) && j < numsteps) {
			// float dt = calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			// uint32_t mip = mip_from_dt(dt, pos, n_grid_size, n_cascades);
			// if (density_grid_occupied_at(pos, density_grid, mip, n_grid_size, n_cascades, griddist_type)) {
			// 	coords_out[j] = { warp_position(pos, aabb, warping_type), warped_dir, warp_dt(dt) };
			// 	++j;
			// 	t += dt;
			// } else {
			// 	uint32_t res = n_grid_size>>mip;
			// 	t = advance_to_next_voxel(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type, pos, ray.d, idir, res);
			// }

			float dt = calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			Vector3f warped_pos = warp_position(pos, aabb, warping_type);
			uint32_t level = level_from_pos(pos, warped_pos, aabb, n_cascades, griddist_type);
			if (density_grid_occupied_at(pos, warped_pos, aabb, density_grid, level, n_grid_size, n_cascades, griddist_type)) {
				coords_out[j] = { warped_pos, warped_dir, warp_dt(dt) };
				++j;
			}
			t += dt;
			if (dt==0) break;
		}
	} else {
		while (aabb.contains(pos = ray.o + t * ray.d) && j < n_max_steps) {
			float dt = calc_dt(t, n_max_steps, cone_angle, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			coords_out[j] = { warp_position(pos, aabb, warping_type), warped_dir, warp_dt(dt) };
			++j;
			t += dt;
			if (dt==0) break;
		}
	}
}

__global__ void init_rays_with_payload_kernel_nerf(
	const Vector2i resolution,
	const Vector2f focal_length,
	const Vector2f principal_point,
	const Matrix<float, 3, 4> camera_matrix,
	const uint32_t spp,
	const bool snap_to_pixel_centers,
	const BoundingBox aabb,
	RayPayload* __restrict__ payloads
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= resolution.x() || y >= resolution.y()) return;

	// generate the ray of a pixel
	Ray ray = pixel_to_ray(
		{x, y},
		resolution,
		focal_length,
		principal_point,
		camera_matrix,
		spp,
		snap_to_pixel_centers
	);

	// determine whether the ray is alive
	const uint32_t idx = x + resolution.x() * y;
	RayPayload& payload = payloads[idx];

	// 0.05f is the nearest distance in rendering
	const float t = fmaxf(aabb.ray_intersect(ray.o, ray.d).x(), 0.05f) + 1e-6f;

	if (aabb.contains(ray.o + ray.d * t)) {
		payload.o = ray.o;
		payload.d = ray.d;
		payload.t = t;
		payload.idx = idx;
		payload.n_steps = 0;

		payload.alive = true;
	} else {
		payload.o = ray.o;

		payload.alive = false;
	}
}

__global__ void advance_pos_nerf(
	const uint32_t n_rays,
	const uint32_t spp,
	RayPayload* __restrict__ payloads,
	EWarpingType warping_type,
	BoundingBox aabb,
	const uint32_t n_grid_size,
	const uint32_t n_cascades,
	EGridDistType griddist_type,
	const uint8_t* __restrict__ density_grid,
	const uint32_t n_max_steps,
	const float cone_angle_constant,
	const float min_cone_stepsize,
	const float max_cone_stepsize,
	const float near_distance,
	const float far_distance,
	ESamplingType sampling_type
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	RayPayload& payload = payloads[i];
	if (!payload.alive) return;

	const Vector3f origin = payload.o;
	const Vector3f dir = payload.d;
	const Vector3f idir = dir.cwiseInverse();

	float t = payload.t;
	float dt = calc_dt(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
	t += ld_random_val(spp, i * 786433) * dt;
	Vector3f pos;

	while (true) {
		if (!aabb.contains(pos = origin + dir * t)) {
			payload.alive = false;
			break;
		}

		// occupancy grid guided
		// dt = calc_dt(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
		// uint32_t mip = mip_from_dt(dt, pos, n_grid_size, n_cascades);
		// if (!density_grid || density_grid_occupied_at(pos, density_grid, mip, n_grid_size, n_cascades, griddist_type)) break;
		// uint32_t res = n_grid_size>>mip;
		// t = advance_to_next_voxel(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type, pos, dir, idir, res);

		// occupancy grid guided
		dt = calc_dt(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
		Vector3f warped_pos = warp_position(pos, aabb, warping_type);
		uint32_t level = level_from_pos(pos, warped_pos, aabb, n_cascades, griddist_type);
		if (!density_grid || density_grid_occupied_at(pos, warped_pos, aabb, density_grid, level, n_grid_size, n_cascades, griddist_type)) break;
		t += dt;
		if (dt==0) {
			payload.alive = false;
			break;
		}
	}

	payload.t = t;
}

__global__ void generate_coords_from_rays_regularly_kernel(
	const uint32_t n_rays,
	const uint32_t n_steps,
	EWarpingType warping_type,
	const BoundingBox aabb,
	const uint32_t n_grid_size,
	const uint32_t n_cascades,
	EGridDistType griddist_type,
	const uint8_t* __restrict__ density_grid,
	const uint32_t n_max_steps,
	const float cone_angle_constant,
	const float min_cone_stepsize,
	const float max_cone_stepsize,
	const float near_distance,
	const float far_distance,
	ESamplingType sampling_type,
	RayPayload* __restrict__ payloads,
	Coordinate* __restrict__ coords
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	RayPayload& payload = payloads[i];
	if (!payload.alive) return;

	const Vector3f origin = payload.o;
	const Vector3f dir = payload.d;
	const Vector3f idir = dir.cwiseInverse();

	float t = payload.t;
	for (uint32_t j = 0; j < n_steps; ++j) {
		Vector3f pos;
		float dt = 0.0f;
		while (true) {
			// the coord is out of region-of-interest
			if (!aabb.contains(pos = origin + dir * t)) {
				payload.n_steps = j;
				return;
			}

			// occupancy grid guided
			// dt = calc_dt(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			// uint32_t mip = mip_from_dt(dt, pos, n_grid_size, n_cascades);
			// if (!density_grid || density_grid_occupied_at(pos, density_grid, mip, n_grid_size, n_cascades, griddist_type)) break;
			// uint32_t res = n_grid_size>>mip;
			// t = advance_to_next_voxel(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type, pos, dir, idir, res);

			// occupancy grid guided
			dt = calc_dt(t, n_max_steps, cone_angle_constant, min_cone_stepsize, max_cone_stepsize, near_distance, far_distance, sampling_type);
			Vector3f warped_pos = warp_position(pos, aabb, warping_type);
			uint32_t mip = level_from_pos(pos, warped_pos, aabb, n_cascades, griddist_type);
			if (!density_grid || density_grid_occupied_at(pos, warped_pos, aabb, density_grid, mip, n_grid_size, n_cascades, griddist_type)) break;
			t += dt;
			if (dt==0) {
				payload.n_steps = j;
				return;
			}
		}

		coords[i*n_steps+j] = { warp_position(pos, aabb, warping_type), warp_direction(dir), warp_dt(dt) };
		t += dt;
	}
	payload.t = t;
	payload.n_steps = n_steps;
}

EWarpingType string_to_warping_type(const std::string& str) {
	if (equals_case_insensitive(str, "NGP")) {
		return EWarpingType::NGP;
	} else if (equals_case_insensitive(str, "MipNeRF360")) {
		return EWarpingType::MipNeRF360;
	} else if (equals_case_insensitive(str, "DONeRF")) {
		return EWarpingType::DONeRF;
	} else if (equals_case_insensitive(str, "Sigmoid")) {
		return EWarpingType::Sigmoid;
	} else if (equals_case_insensitive(str, "ERP")) {
		return EWarpingType::ERP;
	} else if (equals_case_insensitive(str, "SinusoidalProjection")) {
		return EWarpingType::SinusoidalProjection;
	} else {
		throw std::runtime_error{"Unknown warping type."};
	}
}

ESamplingType string_to_sampling_type(const std::string& str) {
	if (equals_case_insensitive(str, "Original")) {
		return ESamplingType::Original;
	} else if (equals_case_insensitive(str, "Disparity")) {
		return ESamplingType::Disparity;
	} else if (equals_case_insensitive(str, "Log")) {
		return ESamplingType::Log;
	} else {
		throw std::runtime_error{"Unknown sampling type."};
	}
}

EGridDistType string_to_griddist_type(const std::string& str) {
	if (equals_case_insensitive(str, "WarpedSpace_Exp")) {
		return EGridDistType::WarpedSpace_Exp;
	} else if (equals_case_insensitive(str, "WarpedSpace_Reciprocal")) {
		return EGridDistType::WarpedSpace_Reciprocal;
	} else if (equals_case_insensitive(str, "MetricSpace_Exp")) {
		return EGridDistType::MetricSpace_Exp;
	} else if (equals_case_insensitive(str, "MetricSpace_Reciprocal")) {
		return EGridDistType::MetricSpace_Reciprocal;
	}else {
		throw std::runtime_error{"Unknown grid distribution type."};
	}
}

OccupancySampler::OccupancySampler(nlohmann::json& config) 
{
	this->warping_type		= string_to_warping_type(config.value("warping_type", "NGP"));

	this->aabb_scale		= config.value("aabb_scale", 1);
	this->aabb = BoundingBox{Vector3f::Constant(0.5f), Vector3f::Constant(0.5f)};
	this->aabb.inflate(0.5f * this->aabb_scale);

	this->use_grid			= config.value("use_grid", true);
	if (this->use_grid) {
		this->init_grid_with_mask 	= config.value("init_grid_with_mask", true);
		this->n_grid_size 			= config.value("grid_size", 128);
		this->n_grid_elements 		= this->n_grid_size * this->n_grid_size * this->n_grid_size;
		this->n_cascades			= config.value("num_cascades", 1);
		this->n_total_elements		= this->n_grid_elements * this->n_cascades;

		this->ema_decay 			= config.value("ema_decay", 0.95f);
		
		this->griddist_type			= string_to_griddist_type(config.value("grid_distribution_type", "Exp"));
		this->density_grid_current.resize(this->n_total_elements);
		this->density_grid.resize(this->n_total_elements);
		this->density_grid_mean.resize(tcnn::reduce_sum_workspace_size(this->n_grid_elements));
		this->density_grid_bitfield.resize(this->n_total_elements / 8);
	}
	
	this->n_max_steps 				= config.value("maximum_marching_steps", 1024);
	this->cone_angle_constant		= config.value("cone_angle_constant", 0.f);
	this->min_cone_stepsize			= config.value("min_cone_stepsize", SQRT3() / this->n_max_steps);
	this->max_cone_stepsize 		= config.value("max_cone_stepsize", this->use_grid ? (SQRT3() * (1<<(this->n_cascades-1)) / this->n_grid_size) : this->min_cone_stepsize);
	this->near_distance				= config.value("near_distance", 0.2f);
	this->far_distance				= config.value("far_distance", 100.f);
	this->sampling_type				= string_to_sampling_type(config.value("sampling_type", "Original"));
	this->snap_to_pixel_centers_in_training	= config.value("snap_to_pixel_centers_in_training", true);
	this->snap_to_pixel_centers_in_rendering = config.value("snap_to_pixel_centers_in_rendering", true);
	this->min_optical_thickness 	= config.value("min_optical_thickness", 0.01f);
}

void OccupancySampler::init(
	const uint32_t n_images,
	const Vector2i& image_resolution,
	const Vector2f& focal_length,
	const Vector2f& principal_point,
	const GPUMemory<Matrix<float, 3, 4>>& camera_matrices,
	const GPUMemory<__half>& images,
	default_rng_t& rng,
	cudaStream_t stream
) {
	this->i_step = 0;
    this->rng = default_rng_t{rng.next_uint()};

	if (this->use_grid && this->init_grid_with_mask && this->warping_type==EWarpingType::NGP) {
		// only cull away empty regions where no camera is looking when the cameras are actually meaningful.
		linear_kernel(mark_untrained_density_grid, 0, stream,
			this->n_total_elements,
			this->n_grid_size,
			this->n_cascades,
			n_images,
			image_resolution,
			focal_length,
			camera_matrices.data(),
			this->aabb_scale,
			this->aabb,
			this->warping_type,
			this->griddist_type,
			this->density_grid.data()
		);
	} else {
		CUDA_CHECK_THROW(cudaMemsetAsync(this->density_grid.data(), 0, this->density_grid.get_bytes(), stream));
	}
}

void OccupancySampler::sample_positions_from_grid(
	const uint32_t n_uniform_positions,
	const uint32_t n_nonuniform_positions,
	GPUMemory<Position>& grid_positions,
	GPUMemory<uint32_t>& grid_indices,
	cudaStream_t stream
) {
	const uint32_t n_total_positions = n_uniform_positions + n_nonuniform_positions;

	grid_positions.enlarge(n_total_positions);
	grid_indices.enlarge(n_total_positions);

	linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
		n_uniform_positions,
		this->rng,
		this->i_step,
		this->warping_type,
		this->aabb_scale,
		this->aabb,
		this->griddist_type,
		this->density_grid.data(),
		grid_positions.data(),
		grid_indices.data(),
		this->n_grid_size,
		this->n_grid_elements,
		this->n_cascades,
		this->min_cone_stepsize,
		-0.01
	);
	this->rng.advance();

	linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
		n_nonuniform_positions,
		this->rng,
		this->i_step,
		this->warping_type,
		this->aabb_scale,
		this->aabb,
		this->griddist_type,
		this->density_grid.data(),
		grid_positions.data() + n_uniform_positions,
		grid_indices.data() + n_uniform_positions,
		this->n_grid_size,
		this->n_grid_elements,
		this->n_cascades,
		this->min_cone_stepsize,
		this->min_optical_thickness
	);
	this->rng.advance();
}

void OccupancySampler::update_grid(
	const uint32_t n_total_positions,
	const uint32_t padded_output_width,
	GPUMemory<precision_t>& grid_densities,
	GPUMemory<uint32_t>& grid_indices,
	EActivation density_activation,
	cudaStream_t stream
) {
	CUDA_CHECK_THROW(cudaMemsetAsync(this->density_grid_current.data(), 0, this->density_grid_current.get_bytes(), stream));

	linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream,
		n_total_positions,
		grid_indices.data(),
		padded_output_width,
		grid_densities.data(),
		density_activation,
		this->n_grid_size,
		this->min_cone_stepsize,
		this->density_grid_current.data()
	);
	
	linear_kernel(ema_grid_samples_nerf, 0, stream,
		this->n_total_elements,
		this->density_grid_current.data(),
		this->ema_decay,
		this->i_step,
		this->density_grid.data()
	);

	++this->i_step;
}

void OccupancySampler::update_grid_bitfield(cudaStream_t stream)
{
	CUDA_CHECK_THROW(cudaMemsetAsync(this->density_grid_mean.data(), 0, sizeof(float), stream));
	const uint32_t n_elements = this->n_grid_elements;
	reduce_sum(
		this->density_grid.data(), 
		[n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, 
		this->density_grid_mean.data(), this->n_grid_elements, stream);

	linear_kernel(grid_to_bitfield, 0, stream, 
		this->n_total_elements / 8,
		this->density_grid.data(), 
		this->density_grid_bitfield.data(),
		this->density_grid_mean.data(),
		this->min_optical_thickness);

	if (this->griddist_type==EGridDistType::WarpedSpace_Exp || this->griddist_type==EGridDistType::MetricSpace_Exp) {
		for (uint32_t level = 1; level < this->n_cascades; ++level) {
			linear_kernel(bitfield_max_pool, 0, stream, 
				this->n_grid_elements / 64,
				this->n_grid_size,
				this->density_grid_bitfield.data() + this->n_grid_elements * (level-1) / 8,
				this->density_grid_bitfield.data() + this->n_grid_elements * level / 8);
		}
	}
}

void OccupancySampler::sample_coords_from_dataset(
	const uint32_t n_rays,
	const uint32_t n_rays_shift,
	const uint32_t n_coords,
	const std::shared_ptr<Dataset> dataset,
	tcnn::GPUMemory<Ray>& rays, 
	tcnn::GPUMemory<Eigen::Array4f>& rays_rgba,
	uint32_t* __restrict__ n_rays_counter,
	tcnn::GPUMemory<uint32_t>& ray_coord_indices,
	tcnn::GPUMemory<Coordinate>& coords,
	uint32_t* __restrict__ n_coords_counter,
	tcnn::default_rng_t rng,
	cudaStream_t stream
) {
	linear_kernel(generate_training_samples_nerf, 0, stream,
		n_rays,
		n_rays_shift,
		n_coords,
		// dataset
		dataset->n_images,
		dataset->image_resolution,
		dataset->focal_length,
		dataset->principal_point,
		dataset->xforms.data(),
		dataset->images.data(),
		dataset->sampling_probability.data(),
		// warping parameters
		this->warping_type,
		// grid
		this->aabb,
		this->n_grid_size,
		this->n_cascades,
		this->griddist_type,
		this->density_grid_bitfield.data(),
		this->use_grid,
		// sampling parameters
		this->snap_to_pixel_centers_in_training,
		this->n_max_steps,
		this->cone_angle_constant,
		this->min_cone_stepsize,
		this->max_cone_stepsize,
		this->near_distance,
		this->far_distance,
		this->sampling_type,
		// output rays & coordinates
		n_rays_counter,
		n_coords_counter,
		rays.data(),
		rays_rgba.data(),
		ray_coord_indices.data(),
		coords.data(),
		rng
	);
}

void OccupancySampler::generate_rays_from_camera_matrix(
	const Vector2i& resolution, 
	const Vector2f& focal_length, 
	const Vector2f& principal_point, 
	const Matrix<float, 3, 4>& camera_matrix, 
	GPUMemory<RayPayload>& payload,
	CudaRenderBuffer& render_buffer, 
	cudaStream_t stream
) {
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { 
		div_round_up((uint32_t)resolution.x(), threads.x), 
		div_round_up((uint32_t)resolution.y(), threads.y), 
		1 };
	init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(
		resolution,
		focal_length,
		principal_point,
		camera_matrix,
		render_buffer.spp(),
		this->snap_to_pixel_centers_in_rendering,
		this->aabb,
		payload.data()
	);

	linear_kernel(advance_pos_nerf, 0, stream,
		resolution.x() * resolution.y(),
		render_buffer.spp(),
		payload.data(),
		this->warping_type,
		this->aabb,
		this->n_grid_size,
		this->n_cascades,
		this->griddist_type,
		this->use_grid ? this->density_grid_bitfield.data() : nullptr,
		this->n_max_steps,
		this->cone_angle_constant,
		this->min_cone_stepsize,
		this->max_cone_stepsize,
		this->near_distance,
		this->far_distance,
		this->sampling_type
	);
}

void OccupancySampler::generate_coords_from_rays(
	const uint32_t n_rays,
	const uint32_t n_steps,
	tcnn::GPUMemory<RayPayload> payloads,
	tcnn::GPUMemory<Coordinate> coords,
	cudaStream_t stream
) {
	linear_kernel(generate_coords_from_rays_regularly_kernel, 0, stream,
		n_rays,
		n_steps,
		this->warping_type,
		this->aabb,
		this->n_grid_size,
		this->n_cascades,
		this->griddist_type,
		this->use_grid ? this->density_grid_bitfield.data() : nullptr,
		this->n_max_steps,
		this->cone_angle_constant,
		this->min_cone_stepsize,
		this->max_cone_stepsize,
		this->near_distance,
		this->far_distance,
		this->sampling_type,
		payloads.data(),
		coords.data()
	);
}