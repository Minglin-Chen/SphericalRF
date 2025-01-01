#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tinylogger/tinylogger.h>

#include <functional>

// Eigen uses __device__ __host__ on a bunch of defaulted constructors.
// This doesn't actually cause unwanted behavior, but does cause NVCC
// to emit this diagnostic.
#if defined(__NVCC__)
#  if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = esa_on_defaulted_function_ignored
#    pragma nv_diag_suppress = unsigned_compare_with_zero
#  else
#    pragma diag_suppress = esa_on_defaulted_function_ignored
#    pragma diag_suppress = unsigned_compare_with_zero
#  endif
#endif
#include <Eigen/Dense>

#if defined(__CUDA_ARCH__)
	#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
		#define PRAGMA_UNROLL _Pragma("unroll")
		#define PRAGMA_NO_UNROLL _Pragma("unroll 1")
	#else
		#define PRAGMA_UNROLL #pragma unroll
		#define PRAGMA_NO_UNROLL #pragma unroll 1
	#endif
#else
	#define PRAGMA_UNROLL
	#define PRAGMA_NO_UNROLL
#endif

#ifdef __NVCC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

using precision_t = tcnn::network_precision_t;

using Vector2i32 = Eigen::Matrix<uint32_t, 2, 1>;
using Vector3i16 = Eigen::Matrix<uint16_t, 3, 1>;
using Vector4i16 = Eigen::Matrix<uint16_t, 4, 1>;
using Vector4i32 = Eigen::Matrix<uint32_t, 4, 1>;

/* *********************************************************************
 * Constant
 * *********************************************************************/
static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t BATCH_SIZE_MULTIPLE = 256;

static constexpr float LOSS_SCALE = 128.f;

/* *********************************************************************
 * Enumerator
 * *********************************************************************/
enum class EActivation : int {
	None,
	ReLU,
	Logistic,
	Exponential,
};

enum class ERenderMode : int {
	AO,
	Shade,
	Normals,
	Depth,
	Distance,
	Cost,
};

enum class EColorSpace : int {
	Linear,
	SRGB,
	VisPosNeg,
};

enum class ETonemapCurve : int {
	Identity,
	ACES,
	Hable,
	Reinhard
};

enum class EWarpingType : int {
	NGP,
	MipNeRF360,
	DONeRF,
	Sigmoid,
	ERP,
	SinusoidalProjection
};

enum class ESamplingType : int {
	Original,	// compatiable with NGP
	Disparity,
	Log
};

enum class EGridDistType : int {
	WarpedSpace_Exp,
	WarpedSpace_Reciprocal,
	MetricSpace_Exp,
	MetricSpace_Reciprocal
};

enum class ERegularizationType : int {
	None,
	Distortion
};

/* *********************************************************************
 * Helpers
 * *********************************************************************/
class ScopeGuard {
public:
    ScopeGuard(const std::function<void()>& callback) : mCallback{callback} {}
    ScopeGuard(std::function<void()>&& callback) : mCallback{std::move(callback)} {}
    ScopeGuard(const ScopeGuard& other) = delete;
    ScopeGuard(ScopeGuard&& other) { mCallback = std::move(other.mCallback); other.mCallback = {}; }
    ~ScopeGuard() { mCallback(); }
private:
    std::function<void()> mCallback;
};

/* *********************************************************************
 * Ray & Coordinate
 * *********************************************************************/
// Training
struct Ray {
	Eigen::Vector3f o;
	Eigen::Vector3f d;
};

struct Position {
	HOST_DEVICE Position(const Eigen::Vector3f& pos, float dt) : p{pos} {}
	Eigen::Vector3f p;
};

struct Direction {
	HOST_DEVICE Direction(const Eigen::Vector3f& dir, float dt) : d{dir} {}
	Eigen::Vector3f d;
};

struct Coordinate {
	HOST_DEVICE Coordinate(const Eigen::Vector3f& pos, const Eigen::Vector3f& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}
	Position pos;
	float dt;
	Direction dir;
};

// Rendering
struct RayPayload {
	bool 			alive;		/* ray hits something in region-of-interest */
	Eigen::Vector3f o;			/* origin of ray */
	Eigen::Vector3f d;			/* normalized direction of ray */
	uint32_t 		idx;		/* pixel location */
	float 			t;			/* marching distance */
	uint32_t 		n_steps;	/* marching step number */
};

struct RaysSoA {
	void enlarge(size_t n_elements) {
		payload.enlarge(n_elements);
		rgba.enlarge(n_elements);
	}
	tcnn::GPUMemory<RayPayload> 		payload;
	tcnn::GPUMemory<Eigen::Array4f> 	rgba;
};