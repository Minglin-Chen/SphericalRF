#pragma onece

#include <json/json.hpp>

#include "internal/utils/common.h"
#include "internal/dataset/nerf_synthetic.h"
#include "internal/sampler/bounding_box.h"


/******************************************************************************************
 * Eigen Conversion
 ******************************************************************************************/
template <typename Derived>
void to_json(nlohmann::json& j, const Eigen::MatrixBase<Derived>& mat) {
	for (int row = 0; row < mat.rows(); ++row) {
		if (mat.cols() == 1) {
			j.push_back(mat(row));
		} else {
			nlohmann::json column = nlohmann::json::array();
			for (int col = 0; col < mat.cols(); ++col) {
				column.push_back(mat(row, col));
			}
			j.push_back(column);
		}
	}
}

template <typename Derived>
void from_json(const nlohmann::json& j, Eigen::MatrixBase<Derived>& mat) {
	for (std::size_t row = 0; row < j.size(); ++row) {
		const auto& jrow = j.at(row);
		if (jrow.is_array()) {
			for (std::size_t col = 0; col < jrow.size(); ++col) {
				const auto& value = jrow.at(col);
				mat(row, col) = value.get<typename Eigen::MatrixBase<Derived>::Scalar>();
			}
		} else {
			mat(row) = jrow.get<typename Eigen::MatrixBase<Derived>::Scalar>();
		}
	}
}

template <typename Derived>
void to_json(nlohmann::json& j, const Eigen::QuaternionBase<Derived>& q) {
	j.push_back(q.w());
	j.push_back(q.x());
	j.push_back(q.y());
	j.push_back(q.z());
}

template <typename Derived>
void from_json(const nlohmann::json& j, Eigen::QuaternionBase<Derived>& q) {
	using Scalar = typename Eigen::QuaternionBase<Derived>::Scalar;
	q.w() = j.at(0).get<Scalar>();
	q.x() = j.at(1).get<Scalar>();
	q.y() = j.at(2).get<Scalar>();
	q.z() = j.at(3).get<Scalar>();
}

/******************************************************************************************
 * NeRFSynthetic Dataset Conversion
 ******************************************************************************************/
inline void to_json(nlohmann::json& j, const NeRFSynthetic& dataset) {
	j["scale"] = dataset.scale;
	to_json(j["offset"], dataset.offset);
	j["n_images"] = dataset.n_images;
	to_json(j["image_resolution"], dataset.image_resolution);
	to_json(j["focal_length"], dataset.focal_length);
	to_json(j["principal_point"], dataset.principal_point);
}

inline void from_json(const nlohmann::json& j, NeRFSynthetic& dataset) {
	dataset.scale = j.at("scale");
	from_json(j.at("offset"), dataset.offset);
	dataset.n_images = j.at("n_images");
	from_json(j.at("image_resolution"), dataset.image_resolution);
	from_json(j.at("focal_length"), dataset.focal_length);
	from_json(j.at("principal_point"), dataset.principal_point);
}

/******************************************************************************************
 * Boundingbox Conversion
 ******************************************************************************************/
inline void to_json(nlohmann::json& j, const BoundingBox& box) {
	to_json(j["min"], box.min);
	to_json(j["max"], box.max);
}

inline void from_json(const nlohmann::json& j, BoundingBox& box) {
	from_json(j.at("min"), box.min);
	from_json(j.at("max"), box.max);
}