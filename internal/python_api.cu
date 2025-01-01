#include <filesystem/path.h>

#include "internal/testbed.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11_json/pybind11_json.hpp>

using namespace tcnn;
using namespace Eigen;
using namespace nlohmann;
namespace py = pybind11;


py::array_t<float> Testbed::render(int width, int height, int spp, bool to_srgb) 
{
	m_rendering_buffer.render_buffer.resize({width, height});
	m_rendering_buffer.render_buffer.reset_accumulation();

	for (int i = 0; i < spp; ++i) {
		render_frame(
			m_rendering_buffer.camera_matrix, 
			m_rendering_buffer.render_buffer, 
			to_srgb);
	}

	py::array_t<float> image({height, width, 4});
	CUDA_CHECK_THROW(cudaMemcpy2DFromArray(
		image.request().ptr, 
		width * 4 * sizeof(float), 
		m_rendering_buffer.render_buffer.surface_provider().array(), 
		0, 
		0, 
		width * 4 * sizeof(float), 
		height, 
		cudaMemcpyDeviceToHost));

	return image;
}

PYBIND11_MODULE(pyngp, m) {
	m.doc() = "Instant neural graphics primitives (NeRF) minimal version";

	py::enum_<EActivation>(m, "Activation")
		.value("None", EActivation::None)
		.value("ReLU", EActivation::ReLU)
		.value("Logistic", EActivation::Logistic)
		.value("Exponential", EActivation::Exponential)
		.export_values();
		
	py::enum_<ERenderMode>(m, "RenderMode")
		.value("AO", ERenderMode::AO)
		.value("Shade", ERenderMode::Shade)
		.value("Normals", ERenderMode::Normals)
		.value("Depth", ERenderMode::Depth)
		.value("Distance", ERenderMode::Distance)
		.value("Cost", ERenderMode::Cost)
		.export_values();

	py::enum_<EColorSpace>(m, "ColorSpace")
		.value("Linear", EColorSpace::Linear)
		.value("SRGB", EColorSpace::SRGB)
		.export_values();

	py::enum_<ETonemapCurve>(m, "TonemapCurve")
		.value("Identity", ETonemapCurve::Identity)
		.value("ACES", ETonemapCurve::ACES)
		.value("Hable", ETonemapCurve::Hable)
		.value("Reinhard", ETonemapCurve::Reinhard)
		.export_values();

	py::enum_<ELossType>(m, "LossType")
		.value("L2", ELossType::L2)
		.value("L1", ELossType::L1)
		.value("Mape", ELossType::Mape)
		.value("Smape", ELossType::Smape)
		.value("SmoothL1", ELossType::SmoothL1)
		.value("LogL1", ELossType::LogL1)
		.value("RelativeL2", ELossType::RelativeL2)
		.export_values();

	py::class_<Testbed> testbed(m, "Testbed");
	testbed
		.def(py::init())
		// Configuration & I/O
		.def("load_model_config", &Testbed::load_model_config, "Reload the network from a config file.")
		.def("save_snapshot", &Testbed::save_snapshot, py::arg("snapshot_path"), py::arg("serialize_optimizer")=false, "Save a snapshot of the currently trained model")
		.def("load_snapshot", &Testbed::load_snapshot, py::arg("snapshot_path"), "Load a previously saved snapshot")
		.def("load_training_data", &Testbed::load_training_data, "Load training data from a given path.")
		// Training
		.def("train", &Testbed::train, "Perform a specified number of training steps.")
		// Rendering
		.def("render", &Testbed::render, "Renders an image at the requested resolution.", py::arg("width")=1920, py::arg("height")=1080, py::arg("spp")=1, py::arg("to_srgb")=true)
		/* Interesting Members */
		.def_readwrite("loss_type", &Testbed::m_loss_type)
		.def_readwrite("density_activation", &Testbed::m_density_activation)
		.def_readwrite("rgb_activation", &Testbed::m_rgb_activation)
		.def_readwrite("background_color", &Testbed::m_background_color)
		.def_readwrite("color_space", &Testbed::m_color_space)
		// Training
		.def_readonly("training_buffer", &Testbed::m_training_buffer)
		// Rendering
		.def_readonly("rendering_buffer", &Testbed::m_rendering_buffer)
		.def_property("fov", &Testbed::fov, &Testbed::set_fov)
		.def_property("fov_xy", &Testbed::fov_xy, &Testbed::set_fov_xy)
		.def("set_nerf_camera_matrix", &Testbed::set_nerf_camera_matrix)
		;

	py::class_<Testbed::TrainingBuffer> training_buffer(testbed, "TrainingBuffer");
	training_buffer
		.def_readonly("n_rays_per_batch", &Testbed::TrainingBuffer::n_rays_per_batch)
		.def_readonly("n_rays_counter", &Testbed::TrainingBuffer::n_rays_counter_cpu)
		.def_readonly("measured_batch_size_before_compaction", &Testbed::TrainingBuffer::measured_batch_size_before_compaction)
		.def_readonly("measured_batch_size", &Testbed::TrainingBuffer::measured_batch_size)
		.def_readonly("loss", &Testbed::TrainingBuffer::loss_scalar)
		.def_readonly("training_prep_ms", &Testbed::TrainingBuffer::training_prep_ms)
		.def_readonly("training_ms", &Testbed::TrainingBuffer::training_ms)
		.def_readonly("i_step", &Testbed::TrainingBuffer::i_step)
		;

	py::class_<Testbed::RenderingBuffer> rendering_buffer(testbed, "RenderingBuffer");
	rendering_buffer
		.def_readwrite("render_mode", &Testbed::RenderingBuffer::render_mode)
		.def_readwrite("tonemap_curve", &Testbed::RenderingBuffer::tonemap_curve)
		.def_readwrite("exposure", &Testbed::RenderingBuffer::exposure)
		.def_readwrite("fov_axis", &Testbed::RenderingBuffer::fov_axis)
		.def_readwrite("relative_focal_length", &Testbed::RenderingBuffer::relative_focal_length)
		.def_readwrite("principal_point", &Testbed::RenderingBuffer::principal_point)
		.def_readwrite("MIN_STEPS_INBETWEEN_COMPACTION", &Testbed::RenderingBuffer::MIN_STEPS_INBETWEEN_COMPACTION)
		.def_readwrite("MAX_STEPS_INBETWEEN_COMPACTION", &Testbed::RenderingBuffer::MAX_STEPS_INBETWEEN_COMPACTION)
		;
}
