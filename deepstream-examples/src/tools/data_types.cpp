#include "data_types.hpp"

Point2D Point2D::operator+(Point2D const& rhs) const
{
    Point2D result = *this;
    result.x += rhs.x;
    result.y += rhs.y;
    return result;
}

Point2D& Point2D::operator+=(Point2D const& rhs)
{
    x += rhs.x;
    y += rhs.y;

    return *this;
}

Point2D Point2D::operator-(Point2D const& rhs) const
{
    Point2D result = *this;
    result.x -= rhs.x;
    result.y -= rhs.y;
    return result;

}

Point2D& Point2D::operator-=(Point2D const& rhs)
{
    x -= rhs.x;
    y -= rhs.y;

    return *this;
}

Point2D Point2D::operator*(Point2D const& rhs) const
{
    Point2D result = *this;
    result.x *= rhs.x;
    result.y *= rhs.y;
    return result;

}

Point2D& Point2D::operator*=(Point2D const& rhs)
{
    x *= rhs.x;
    y *= rhs.y;

    return *this;
}

Point2D Point2D::operator/(Point2D const& rhs) const
{
    Point2D result = *this;
    result.x /= rhs.x;
    result.y /= rhs.y;
    return result;

}

Point2D& Point2D::operator/=(Point2D const& rhs)
{
    x /= rhs.x;
    y /= rhs.y;

    return *this;
}