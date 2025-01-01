#include "internal/utils/loss.h"

#ifdef copysign
#undef copysign
#endif

using namespace Eigen;
using namespace tcnn;

inline __device__ Array3f copysign(const Array3f& a, const Array3f& b) {
	return {
		copysignf(a.x(), b.x()),
		copysignf(a.y(), b.y()),
		copysignf(a.z(), b.z()),
	};
}

inline __device__ LossAndGradient l2_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	return {
		difference * difference,
		2.0f * difference
	};
}

inline __device__ LossAndGradient relative_l2_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f factor = (prediction * prediction + Array3f::Constant(1e-2f)).inverse();
	return {
		difference * difference * factor,
		2.0f * difference * factor
	};
}

inline __device__ LossAndGradient l1_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	return {
		difference.abs(),
		copysign(Array3f::Ones(), difference),
	};
}

inline __device__ LossAndGradient smooth_l1_loss(const Array3f& target, const Array3f& prediction, float alpha = 1) {
	Array3f difference = prediction - target;
	Array3f abs_diff = difference.abs();
	Array3f square = 0.5f/alpha * difference * difference;
	return {
		{
			abs_diff.x() > alpha ? (abs_diff.x() - 0.5f * alpha) : square.x(),
			abs_diff.y() > alpha ? (abs_diff.y() - 0.5f * alpha) : square.y(),
			abs_diff.z() > alpha ? (abs_diff.z() - 0.5f * alpha) : square.z(),
		},
		{
			abs_diff.x() > alpha ? (difference.x() > 0 ? 1.0f : -1.0f) : (difference.x() / alpha),
			abs_diff.y() > alpha ? (difference.y() > 0 ? 1.0f : -1.0f) : (difference.y() / alpha),
			abs_diff.z() > alpha ? (difference.z() > 0 ? 1.0f : -1.0f) : (difference.z() / alpha),
		},
	};
}

inline __device__ LossAndGradient log_l1_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f divisor = difference.abs() + Array3f::Ones();
	return {
		divisor.log(),
		copysign(divisor.inverse(), difference),
	};
}

inline __device__ LossAndGradient smape_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f factor = (0.5f * (prediction.abs() + target.abs()) + Array3f::Constant(1e-2f)).inverse();
	return {
		difference.abs() * factor,
		copysign(factor, difference),
	};
}

inline __device__ LossAndGradient mape_loss(const Array3f& target, const Array3f& prediction) {
	Array3f difference = prediction - target;
	Array3f factor = (prediction.abs() + Array3f::Constant(1e-2f)).inverse();
	return {
		difference.abs() * factor,
		copysign(factor, difference),
	};
}

__device__ LossAndGradient loss_and_gradient(const Vector3f& target, const Vector3f& prediction, ELossType loss_type) {
	switch (loss_type) {
		case ELossType::RelativeL2:  return relative_l2_loss(target, prediction); break;
		case ELossType::L1:          return l1_loss(target, prediction); break;
		case ELossType::Mape:        return mape_loss(target, prediction); break;
		case ELossType::Smape:       return smape_loss(target, prediction); break;
		case ELossType::SmoothL1:    return smooth_l1_loss(target, prediction, 0.1f); break;
		case ELossType::LogL1:       return log_l1_loss(target, prediction); break;
		default: case ELossType::L2: return l2_loss(target, prediction); break;
	}
}

ELossType string_to_loss_type(const std::string& str) {
	if (equals_case_insensitive(str, "L2")) {
		return ELossType::L2;
	} else if (equals_case_insensitive(str, "RelativeL2")) {
		return ELossType::RelativeL2;
	} else if (equals_case_insensitive(str, "L1")) {
		return ELossType::L1;
	} else if (equals_case_insensitive(str, "Mape")) {
		return ELossType::Mape;
	} else if (equals_case_insensitive(str, "Smape")) {
		return ELossType::Smape;
	} else if (equals_case_insensitive(str, "SmoothL1")) {
		return ELossType::SmoothL1;
	} else if (equals_case_insensitive(str, "LogL1")) {
		return ELossType::LogL1;
	} else {
		throw std::runtime_error{"Unknown loss type."};
	}
}