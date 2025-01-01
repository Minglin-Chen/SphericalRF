#pragma once

#include "internal/utils/common.h"

#include <tiny-cuda-nn/gpu_memory.h>

#include <memory>
#include <vector>


class SurfaceProvider {
public:
	virtual cudaSurfaceObject_t surface() = 0;
	virtual cudaArray_t array() = 0;
	virtual Eigen::Vector2i resolution() const = 0;
	virtual void resize(const Eigen::Vector2i&) = 0;
};


class CudaSurface2D : public SurfaceProvider {
public:
	CudaSurface2D() {
		m_array = nullptr;
		m_surface = 0;
	}

	~CudaSurface2D() {
		free();
	}

	void free();

	void resize(const Eigen::Vector2i& size) override;

	cudaSurfaceObject_t surface() override {
		return m_surface;
	}

	cudaArray_t array() override {
		return m_array;
	}

	Eigen::Vector2i resolution() const override {
		return m_size;
	}

private:
	Eigen::Vector2i m_size = Eigen::Vector2i::Constant(0);
	cudaArray_t m_array;
	cudaSurfaceObject_t m_surface;
};


class CudaRenderBuffer {
public:
	CudaRenderBuffer(const std::shared_ptr<SurfaceProvider>& surf) : m_surface_provider{surf} {}

	CudaRenderBuffer(const CudaRenderBuffer& other) = delete;
	CudaRenderBuffer(CudaRenderBuffer&& other) = default;

	cudaSurfaceObject_t surface() {
		return m_surface_provider->surface();
	}

	Eigen::Vector2i resolution() const {
		return m_surface_provider->resolution();
	}

	void resize(const Eigen::Vector2i& size);

	void reset_accumulation() {
		m_spp = 0;
	}

	uint32_t spp() const {
		return m_spp;
	}

	Eigen::Array4f* frame_buffer() const {
		return m_frame_buffer.data();
	}

	Eigen::Array4f* accumulate_buffer() const {
		return m_accumulate_buffer.data();
	}

	void clear_frame_buffer(cudaStream_t stream);

	void accumulate(cudaStream_t stream);

	void tonemap(float exposure, const Eigen::Array4f& background_color, EColorSpace output_color_space, cudaStream_t stream);

	SurfaceProvider& surface_provider() {
		return *m_surface_provider;
	}

	void set_color_space(EColorSpace color_space) {
		if (color_space != m_color_space) {
			m_color_space = color_space;
			reset_accumulation();
		}
	}

	void set_tonemap_curve(ETonemapCurve tonemap_curve) {
		if (tonemap_curve != m_tonemap_curve) {
			m_tonemap_curve = tonemap_curve;
			reset_accumulation();
		}
	}

private:
	uint32_t m_spp = 0;
	EColorSpace m_color_space = EColorSpace::Linear;
	ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;

	tcnn::GPUMemory<Eigen::Array4f> m_frame_buffer;
	tcnn::GPUMemory<Eigen::Array4f> m_accumulate_buffer;

	std::shared_ptr<SurfaceProvider> m_surface_provider;
};
