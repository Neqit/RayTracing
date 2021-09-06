#ifndef VECTOR3_H
#define VECTOR3_H

#include <cmath>
#include <iostream>
#include <random>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


inline float random_float() {
	static std::uniform_real_distribution<float> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline float random_float(float min, float max) {
	static std::uniform_real_distribution<float> distribution(min, max);
	static std::mt19937 generator;
	return distribution(generator);
}


class Vector3
{
public:
	__host__ __device__ Vector3() : e{ 0,0,0 } {}
	__host__ __device__ Vector3(float e0, float e1, float e2) : e{ e0,e1,e2 } {}

	__host__ __device__ float x() const { return e[0]; }
	__host__ __device__ float y() const { return e[1]; }
	__host__ __device__ float z() const { return e[2]; }

	__host__ __device__ Vector3 operator-() const { return Vector3(-e[0], -e[1], -e[2]); }
	__host__ __device__ float operator[](int i) const { return e[i]; }
	__host__ __device__ float& operator[](int i) { return e[i]; }

	__host__ __device__ Vector3& operator+=(const Vector3& vec) {
		e[0] += vec.e[0];
		e[1] += vec.e[1];
		e[2] += vec.e[2];

		return *this;
	}

	__host__ __device__ Vector3& operator*=(const float mult) {
		e[0] *= mult;
		e[1] *= mult;
		e[2] *= mult;
		return *this;
	}

	__host__ __device__ Vector3& operator/=(const float div) {
		return *this *= 1 / div;
	}

	__host__ __device__ float length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	__host__ __device__ float length() const {
		return std::sqrt(length_squared());
	}

	inline static Vector3 random() {
		return Vector3(random_float(), random_float(), random_float());
	}

	inline static Vector3 random(float min, float max) {
		return Vector3(random_float(min, max), random_float(min, max), random_float(min, max));
	}

	//Returns true whenever one of the vector dimesnsons close to 0
	bool near_zero() const {
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}


public:
	float e[3];

};

using point3 = Vector3;   // 3D point
using color = Vector3;    // RGB color



inline std::ostream& operator<<(std::ostream& out, const Vector3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline Vector3 operator+(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline Vector3 operator-(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline Vector3 operator*(float t, const Vector3& v) {
	return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vector3 operator*(const Vector3& v, float t) {
	return t * v;
}

__host__ __device__ inline Vector3 operator/(Vector3 v, float t) {
	return (1 / t) * v;
}

__host__ __device__ inline float dot(const Vector3& u, const Vector3& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

__host__ __device__ inline Vector3 cross(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline Vector3 normalize(Vector3 v) {
	return v / v.length();
}


inline Vector3 random_in_unit_sphere() {
	while (true) {
		auto p = Vector3::random(-1, 1);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

inline Vector3 random_unit_vector() {
	return normalize(random_in_unit_sphere());
}

__host__ __device__ inline Vector3 reflect(const Vector3& v, const Vector3& n) {
	return v - 2 * dot(v, n) * n;
}

Vector3 random_in_unit_disk() {
	while (true) {
		auto p = Vector3(random_float(-1, 1), random_float(-1, 1), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}


#endif