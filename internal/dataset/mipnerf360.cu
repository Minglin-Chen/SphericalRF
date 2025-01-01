#include <filesystem/path.h>
#include <json/json.hpp>

#include <stb_image/stb_image.h>

#include "internal/utils/common_device.h"
#include "internal/dataset/mipnerf360.h"
#include "internal/dataset/thread_pool.h"


void MipNeRF360::load_from_json(
	const std::string& path, 
	const float scale,
	const Eigen::Vector3f offset,
	const bool discard_saturation) 
{
	const filesystem::path jsonpath = path;

	tlog::info() << "Loading dataset from " << jsonpath.str();
	const filesystem::path basepath = jsonpath.parent_path();

	this->scale = scale;
	this->offset = offset;

	// 0. load json
	std::ifstream f{ jsonpath.str() };
	nlohmann::json json = nlohmann::json::parse(f, nullptr, true, true);
	auto frames = json["frames"];
	this->n_images = json["frames"].size();

	// 1. load image and pose
	std::vector<uint8_t*> images(this->n_images, nullptr);
	std::vector<Eigen::Matrix<float, 3, 4>> xforms(this->n_images);
	this->filenames.resize(this->n_images);
	
	auto progress = tlog::progress(this->n_images);
	std::atomic<int> n_loaded{0};
	ThreadPool pool;
	pool.parallelFor<size_t>(0, this->n_images, [&](size_t i) {
		const auto& frame = frames[i];
		auto& 		filename = this->filenames[i];
		auto& 		image = images[i];
		auto& 		xform = xforms[i];

		// 1.1 load image
		filesystem::path path = basepath / frame["file_path"];
		if (path.extension() == "") {
			path = path.with_extension("png");
			if (!path.exists()) {
				path = path.with_extension("jpg");
			}
			if (!path.exists()) {
				throw std::runtime_error{ "Could not find image file: " + path.str()};
			}
		}
		filename = path.filename();

		Eigen::Vector2i res = Eigen::Vector2i::Zero();
		int comp = 0;
		image = stbi_load(path.str().c_str(), &res.x(), &res.y(), &comp, 4);

		if (!image) {
			throw std::runtime_error{ "image not found: " + path.str() };
		}
		if (!this->image_resolution.isZero() && res!=this->image_resolution) {
			throw std::runtime_error{ "training images are not all the same size" };
		}
		// get image resolution
		this->image_resolution = res;

		// 1.2 get pose matrix (camera to world)
		for (int m = 0; m < 3; ++m) {
			for (int n = 0; n < 4; ++n) {
				xform(m, n) = float(frame["transform_matrix"][m][n]);
			}
		}
		xform = spec_opengl_to_opencv(xform, this->scale, this->offset);

		// update
		progress.update(++n_loaded);
	});

	tlog::success() << "Loaded " << images.size() 
					<< " images of size " << this->image_resolution.x() << "x" << this->image_resolution.y() 
					<< " after " << tlog::durationToString(progress.duration());

	// 2. load focal length & principal point
	float fx = json["fx"];
	float fy = json["fy"];
	float cx = json["cx"];
	float cy = json["cy"];
	this->focal_length = Eigen::Vector2f{fx, fy};
	this->principal_point = Eigen::Vector2f{cx, cy};
	
	// 3. copy image data from CPU to GPU
	size_t n_pixels = this->image_resolution.prod();
	size_t image_size = n_pixels * 4;
	// 3.1 copy to a temporary GPU buffer
	tcnn::GPUMemory<uint8_t> images_gpu(image_size * this->n_images);
	pool.parallelFor<size_t>(0, this->n_images, [&](size_t i) {
		CUDA_CHECK_THROW(cudaMemcpy(images_gpu.data() + image_size*i, images[i], image_size, cudaMemcpyHostToDevice));
		free(images[i]);
	});
	// 3.2 cast on the GPU
	this->images.resize(image_size * this->n_images);
	tcnn::linear_kernel(from_rgba32<__half>, 0, nullptr, 
		n_pixels * this->n_images,
		images_gpu.data(), 
		this->images.data()
	);
	// 3.3 initialize the sampling probability
	this->sampling_probability.resize(n_pixels * this->n_images);
	this->sampling_probability.memset((int)(1.0f));
	if (discard_saturation) {
		tcnn::linear_kernel(discard_saturated_pixels<__half>, 0, nullptr, 
			n_pixels * this->n_images,
			images_gpu.data(), 
			this->sampling_probability.data()
		);
	}
	
	// 4. copy pose data from CPU to GPU
	this->xforms.resize_and_copy_from_host(xforms);

	// synchronize
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
}