#include "internal/render/ray_marcher.h"

#include "internal/utils/common_device.h"

using namespace Eigen;
using namespace tcnn;


__global__ void volume_rendering_with_loss_and_gradient_kernal(
	const uint32_t n_rays,
	const uint32_t n_max_compacted_coords,
	// network input (ray & coordinate)
	const Ray* __restrict__ rays,
	const Array4f* __restrict__ rays_rgba,
	const uint32_t* __restrict__ n_rays_counter,
	uint32_t* __restrict__ ray_coord_indices,
	const Coordinate* __restrict__ coords,
	Coordinate* __restrict__ compacted_coords,
	uint32_t* __restrict__ n_compacted_coords_counter,
	// network output (density & color) with gradient
	const precision_t* network_output,
	precision_t* dloss_doutput,
	const int padded_output_width,
	EActivation rgb_activation,
	EActivation density_activation,
	// loss
	float* __restrict__ loss_output,
	float* __restrict__ reg_distortion_buffer,
	ELossType loss_type,
	float loss_scale,
	ERegularizationType regularization_type,
	// sampling parameters
	BoundingBox aabb,
	const uint32_t n_cascades,
	const float min_cone_stepsize,
	const float* __restrict__ mean_density_ptr,
	const float min_optical_thickness,
	const bool use_grid,
	// others
	Array3f background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_color,
	const float transmittance_threshold,
	default_rng_t rng
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *n_rays_counter) return;

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps 	= ray_coord_indices[i*2+0];
	uint32_t base 		= ray_coord_indices[i*2+1];

	coords += base;
	network_output += base * 4;

	// 1. volume rendering forward
	float T = 1.f;

	Array3f rgb_ray = Array3f::Zero();

	uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < transmittance_threshold) break;

		const vector_t<precision_t, 4> local_network_output = *(vector_t<precision_t, 4>*)network_output;
		const float 	density = network_to_density(float(local_network_output[3]), density_activation);
		const Array3f 	rgb 	= network_to_rgb(local_network_output, rgb_activation);
		const float 	dt 		= unwarp_dt(coords->dt);

		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray += weight * rgb;

		T *= (1.f - alpha);

		network_output += 4;
		coords++;
	}

	// background color
	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	Array3f pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// get the ground-truth RGB
	const Array4f texsamp = rays_rgba[i];
	Array3f rgbtarget;
	if (train_in_linear_color || color_space == EColorSpace::Linear) {
		rgbtarget = texsamp.head<3>() + (1 - texsamp.w()) * background_color;

		if (!train_in_linear_color) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.w() > 0) {
			rgbtarget = linear_to_srgb(texsamp.head<3>() / texsamp.w()) * texsamp.w() + (1 - texsamp.w()) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

	// step again, this time computing loss
	// rewind the pointer
	network_output -= 4 * compacted_numsteps; 
	coords -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(n_compacted_coords_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(n_max_compacted_coords - min(n_max_compacted_coords, compacted_base), compacted_numsteps);
	ray_coord_indices[i*2+0] = compacted_numsteps;
	ray_coord_indices[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	compacted_coords 	+= compacted_base;
	dloss_doutput 		+= compacted_base * padded_output_width;

	// compute loss and dloss/dfinal_rgb
	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);

	float mean_loss = lg.loss.mean();
	if (loss_output) { loss_output[i] = mean_loss / (float)n_rays; }

	loss_scale /= n_rays;

	const float output_l2_reg = rgb_activation == EActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = (use_grid && (*mean_density_ptr < min_optical_thickness)) ? 1e-4f : 0.0f;

	// pre-compute the summation accroding to DVGO v2
	T = 1.f;
	float sum_weight = 0.f;
	float sum_weight_pos = 0.f;
	if (regularization_type == ERegularizationType::Distortion) {
		reg_distortion_buffer += compacted_base * 2;

		for (uint32_t j = 0; j < compacted_numsteps; ++j) {
			float dt = unwarp_dt(coords[j].dt);
			const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
			const float density = network_to_density(float(local_network_output[3]), density_activation);
			const float alpha = 1.f - __expf(-density * dt);
			const float weight = alpha * T;

			// sum of w
			sum_weight += weight;
			// sum of w*m
			sum_weight_pos += weight * (j+1) / compacted_numsteps;

			reg_distortion_buffer[0] = sum_weight;
			reg_distortion_buffer[1] = sum_weight_pos;

			reg_distortion_buffer 	+= 2;
			network_output 			+= 4;
		}

		reg_distortion_buffer 	-= 2 * compacted_numsteps;
		network_output 			-= 4 * compacted_numsteps;
	}

	// now do it again computing gradients
	Array3f rgb_ray2 = { 0.f,0.f,0.f };
	T = 1.f;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {

		// compact network inputs
		compacted_coords[j] = coords[j];
		float dt = unwarp_dt(coords[j].dt);
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray2 += weight * rgb;
		T *= (1.f - alpha);

		tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;
		// chain rule to go from dloss/drgb to dloss/dmlp_output, regularization is used to penalize too large color values
		const Array3f dloss_by_drgb = weight * lg.gradient;
		local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); 
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const Array3f suffix = rgb_ray - rgb_ray2;
		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		float dloss_by_dmlp = density_derivative * (dt * lg.gradient.matrix().dot((T * rgb - suffix).matrix()));
		local_dL_doutput[3] = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0 ? -output_l1_reg_density : 0.0f);

		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += 4;
	}
}

__global__ void volume_rendering_kernel(
	const uint32_t n_rays,
	const uint32_t n_steps,
	const uint32_t n_steps_offset,
	RayPayload* __restrict__ payloads,
	Array4f* __restrict__ rgba,
	const Coordinate* __restrict__ network_input,
	const precision_t* __restrict__ network_output,
	const uint32_t padded_output_width,
	const EActivation density_activation,
	const EActivation rgb_activation,
	const ERenderMode render_mode,
	EWarpingType warping_type,
	const BoundingBox aabb,
	const uint32_t n_cascades,
	const float min_cone_stepsize,
	const Matrix<float, 3, 4> camera_matrix,
	const float depth_scale,
	const float transmittance_threshold
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	RayPayload& payload = payloads[i];
	if (!payload.alive) return;

	Vector3f origin = payload.o;
	Vector3f cam_fwd = camera_matrix.col(2);

	Array4f local_rgba = rgba[i];

	uint32_t j = 0; 
	for (; j < payload.n_steps; ++j) {
		// position
		const Vector3f 	warped_pos	= network_input[i*n_steps+j].pos.p;
		const Vector3f 	pos 		= unwarp_position(warped_pos, aabb, warping_type);
		const float 	dt 			= unwarp_dt(network_input[i*n_steps+j].dt);

		const vector_t<precision_t, 4> local_network_output = *(vector_t<precision_t, 4>*)&network_output[(i*n_steps+j)*padded_output_width];
		// density
		float T = 1.f - local_rgba.w();
		float alpha = 1.f - __expf(-network_to_density(float(local_network_output[3]), density_activation) * dt);
		float weight = alpha * T;
		// color
		Array3f rgb = network_to_rgb(local_network_output, rgb_activation);

		if (render_mode == ERenderMode::Normals) {
			// Network input contains the gradient of the network output w.r.t. input.
			// So to compute density gradients, we need to apply the chain rule.
			// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
			Vector3f normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
			rgb = normal.normalized().array();
		} else if (render_mode == ERenderMode::Depth) {
			float z=cam_fwd.dot(pos-origin) * depth_scale;
			rgb = {z,z,z};
		} else if (render_mode == ERenderMode::Distance) {
			float z=(pos-origin).norm() * depth_scale;
			rgb = {z,z,z};
		} else if (render_mode == ERenderMode::AO) {
			rgb = Array3f::Constant(alpha);
		}

		// update
		local_rgba.head<3>() 	+= rgb * weight;
		local_rgba.w() 			+= weight;

		if (local_rgba.w() > (1.0f - transmittance_threshold)) {
			rgba[i] = local_rgba / local_rgba.w();
			break;
		}
	}

	if (j < n_steps) {
		payload.alive = false;
		payload.n_steps = j + n_steps_offset;
	}

	rgba[i] = local_rgba;
}

ERegularizationType string_to_regularization_type(const std::string& str) {
	if (equals_case_insensitive(str, "None")) {
		return ERegularizationType::None;
	} else if (equals_case_insensitive(str, "Distortion")) {
		return ERegularizationType::Distortion;
	} else {
		throw std::runtime_error{"Unknown regularization type."};
	}
}

RayMarcher::RayMarcher(nlohmann::json& config) 
{
	this->train_with_random_bg_color 	= config.value("train_with_random_bg_color", true);
	this->train_in_linear_color 		= config.value("train_in_linear_color", false);
	this->transmittance_threshold 		= config.value("transmittance_threshold", 0.0001f);

	this->regularization_type			= string_to_regularization_type(config.value("regularization_type", "None"));
}

void RayMarcher::volume_rendering_with_loss_and_gradient(
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
	Array3f background_color,
	EColorSpace color_space,
	tcnn::default_rng_t rng,
	cudaStream_t stream
) {
	if (this->regularization_type == ERegularizationType::Distortion) {
		this->reg_distortion_buffer.enlarge(n_max_compacted_coords * 2);
	}

	linear_kernel(volume_rendering_with_loss_and_gradient_kernal, 0, stream,
		n_rays,
		n_max_compacted_coords,
		rays.data(),
		rays_rgba.data(),
		n_rays_counter,
		ray_coord_indices.data(),
		coords.data(),
		compacted_coords.data(),
		n_compacted_coords_counter,
		network_output.data(),
		dloss_doutput.data(),
		padded_output_width,
		rgb_activation,
		density_activation,
		loss_output,
		this->reg_distortion_buffer.data(),
		loss_type,
		loss_scale,
		regularization_type,
		aabb,
		n_cascades,
		min_cone_stepsize,
		density_grid_mean.data(),
		min_optical_thickness,
		use_grid,
		background_color,
		color_space,
		this->train_with_random_bg_color,
		this->train_in_linear_color,
		this->transmittance_threshold,
		rng
	);
}

void RayMarcher::volume_rendering(
	const uint32_t n_rays,
	const uint32_t n_steps,
	const uint32_t i_steps_offset,
	RaysSoA& rays,
	const GPUMemory<Coordinate>& coords,
	const GPUMemory<precision_t>& rgbsigma,
	const uint32_t padded_output_width,
	const EActivation density_activation,
	const EActivation rgb_activation,
	const ERenderMode render_mode,
	EWarpingType warping_type,
	const BoundingBox aabb,
	const uint32_t n_cascades,
	const float min_cone_stepsize,
	const Matrix<float, 3, 4>& camera_matrix,
	const float depth_scale,
	cudaStream_t stream
) {
	linear_kernel(volume_rendering_kernel, 0, stream,
		n_rays,
		n_steps,
		i_steps_offset,
		rays.payload.data(),
		rays.rgba.data(),
		coords.data(),
		rgbsigma.data(),
		padded_output_width,
		density_activation,
		rgb_activation,
		render_mode,
		warping_type,
		aabb,
		n_cascades,
		min_cone_stepsize,
		camera_matrix,
		depth_scale,
		this->transmittance_threshold
	);
}