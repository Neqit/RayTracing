#ifndef RAY_H
#define RAY_H

#include "Vector3.h"

class ray {
public:
    ray() {}
    ray(const point3& origin, const Vector3& direction)
        : orig(origin), dir(direction)
    {}

    point3 origin() const { return orig; }
    Vector3 direction() const { return dir; }

    point3 at(double t) const {
        return orig + t * dir;
    }

public:
    point3 orig;
    Vector3 dir;
};



#endif // !RAY_H

