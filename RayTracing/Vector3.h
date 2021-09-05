#ifndef VECTOR3_H
#define VECTOR3_H

#include <cmath>
#include <iostream>
#include <random>

inline double random_double() {
	static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline double random_double(double min, double max) {
	static std::uniform_real_distribution<double> distribution(min, max);
	static std::mt19937 generator;
	return distribution(generator);
}


class Vector3
{
public:
	Vector3() : e{ 0,0,0 } {}
	Vector3(double e0, double e1, double e2) : e{ e0,e1,e2 } {}
	~Vector3();

	double x() const { return e[0]; }
	double y() const { return e[1]; }
	double z() const { return e[2]; }

	Vector3 operator-() const { return Vector3(-e[0], -e[1], -e[2]); }
	double operator[](int i) const { return e[i]; }
	double& operator[](int i) { return e[i]; }

	Vector3& operator+=(const Vector3& vec) {
		e[0] += vec.e[0];
		e[1] += vec.e[1];
		e[2] += vec.e[2];

		return *this;
	}

	Vector3& operator*=(const double mult) {
		e[0] *= mult;
		e[1] *= mult;
		e[2] *= mult;
		return *this;
	}

	Vector3& operator/=(const double div) {
		return *this *= 1 / div;
	}

	double length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	double length() const {
		return std::sqrt(length_squared());
	}

	inline static Vector3 random() {
		return Vector3(random_double(), random_double(), random_double());
	}

	inline static Vector3 random(double min, double max) {
		return Vector3(random_double(min, max), random_double(min, max), random_double(min, max));
	}


public:
	double e[3];

};

using point3 = Vector3;   // 3D point
using color = Vector3;    // RGB color

Vector3::~Vector3()
{
	//delete[] e;
}

inline std::ostream& operator<<(std::ostream& out, const Vector3& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline Vector3 operator+(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline Vector3 operator-(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline Vector3 operator*(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline Vector3 operator*(double t, const Vector3& v) {
	return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline Vector3 operator*(const Vector3& v, double t) {
	return t * v;
}

inline Vector3 operator/(Vector3 v, double t) {
	return (1 / t) * v;
}

inline double dot(const Vector3& u, const Vector3& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

inline Vector3 cross(const Vector3& u, const Vector3& v) {
	return Vector3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline Vector3 normalize(Vector3 v) {
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




#endif