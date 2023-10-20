#pragma once

#include <vector>
#include <array>

/**
 * @class Point2D
 * @brief A struct to hold 2D points.
 */
struct alignas(float) Point2D
{
    Point2D() : x(0.0f), y(0.0f)
    {}

    Point2D(float x_in, float y_in) : x(x_in), y(y_in)
    {}

    Point2D operator+(Point2D const& rhs) const;
    Point2D& operator+=(Point2D const& rhs);

    Point2D operator-(Point2D const& rhs) const;
    Point2D& operator-=(Point2D const& rhs);

    Point2D operator*(Point2D const& rhs) const;
    Point2D& operator*=(Point2D const& rhs);

    Point2D operator/(Point2D const& rhs) const;
    Point2D& operator/=(Point2D const& rhs);

    float x;
    float y;
};

/**
 * @struct Bbox
 * @brief A struct that defines a bounding box.
 */
struct alignas(float) Bbox
{
    Bbox() : top_left(Point2D()), bottom_right(Point2D())
    {}

    Bbox(Point2D top_left_in, Point2D bottom_right_in) : top_left(top_left_in), bottom_right(bottom_right_in)
    {}

    /**
     * @brief Returns area of the bbox.
     * @return area of the bouding box.
     */
    float getArea() const
    {
        return (bottom_right.x - top_left.x) * (bottom_right.y - top_left.y);
    }

    /**
     * @brief Returns width of the bounding box.
     * @return width of the bouding box.
     */
    float getWidth() const
    {
        return bottom_right.x - top_left.x;
    }

    /**
     * @brief Returns height of the bbox.
     * @return height of the bouding box.
     */
    float getHeight() const
    {
        return bottom_right.x - top_left.y;
    }

    Point2D top_left;
    Point2D bottom_right;
};

/**
 * @struct Landmark
 * @brief A struct that defines landmarks
 */
template <unsigned int N>
struct alignas(float) Landmark
{
    Landmark<N> operator+(Point2D const& rhs) const;
    Landmark<N>& operator+=(Point2D const& rhs);

    Landmark<N> operator-(Point2D const& rhs) const;
    Landmark<N>& operator-=(Point2D const& rhs);

    Landmark<N> operator*(Point2D const& rhs) const;
    Landmark<N>& operator*=(Point2D const& rhs);

    Landmark<N> operator/(Point2D const& rhs) const;
    Landmark<N>& operator/=(Point2D const& rhs);
    
    static constexpr unsigned int size = N;
    Point2D point[N];
};

/**
 * @brief A landmark with 5 points.
 */
using Landmark5 = Landmark<5>;

/**
 * @struct Color
 * @brief A struct that holds RGBA colors
*/
struct alignas(float) Color
{
    constexpr Color(float red_, float green_, float blue_, float alpha_) : red(red_), green(green_), blue(blue_), alpha(alpha_)
    {}

    float red;
    float green;
    float blue;
    float alpha;
};

/**
 * @brief An array of 10 clearly distinguishable colors
*/
static constexpr std::array<Color, 10> TenColours =
    {
        Color{230.f / 255.f, 25.f / 255.f, 75.f / 255.f, 255.f / 255.f},
        Color{245.f / 255.f, 130.f / 255.f, 48.f / 255.f, 255.f / 255.f},
        Color{255.f / 255.f, 255.f / 255.f, 25.f / 255.f, 255.f / 255.f},
        Color{210.f / 255.f, 245.f / 255.f, 60.f / 255.f, 255.f / 255.f},
        Color{60.f / 255.f, 180.f / 255.f, 75.f / 255.f, 255.f / 255.f},
        Color{70.f / 255.f, 240.f / 255.f, 240.f / 255.f, 255.f / 255.f},
        Color{0.f / 255.f, 130.f / 255.f, 200.f / 255.f, 255.f / 255.f},
        Color{145.f / 255.f, 30.f / 255.f, 180.f / 255.f, 255.f / 255.f},
        Color{240.f / 255.f, 50.f / 255.f, 230.f / 255.f, 255.f / 255.f},
        Color{128.f / 255.f, 128.f / 255.f, 128.f / 255.f, 255.f / 255.f}};

/**
 * @brief Get color from a std-container that has random access iterator
 * @param[in] container : a container that defines the colors
 * @param[in] index : an index pointing to the color. A value larger than the 
 * number of colors in the container just starts from the beginning.
 * 
 * Example:
 * Colour myColor = getColour(TenColours, 1);
*/
template <template <typename, std::size_t...> class Container, typename T = Color, std::size_t... INTS>
Color getColour(Container<T, INTS...> const &container, std::size_t index)
{
    return container[index % container.size()];
}

/**
 * @brief Get a color from a list of 10 different colors
 * @param[in] index : an index pointing to the color. A value >= 10 just
 * fetches colors from the beginning
*/
static Color getColour(std::size_t index)
{
    return TenColours.at(index % TenColours.size());
}

/**
 * @struct Bbox
 * @brief A struct that defines the class confidences/probabilities.
 */
struct alignas(float) Probs
{
    float class1_confidence;
    float class2_confidence;
};

/**
 * @struct IndexWithProbability
 * @brief A struct that contains index and a confidence/probability score.
 */
struct IndexWithProbability
{
    IndexWithProbability() : index(0), probability(-1.0f)
    {
    }

    IndexWithProbability(std::size_t index_in, float probability_in) : index(index_in), probability(probability_in)
    {
    }

    std::size_t index;
    float probability;
};

template<unsigned int N>
Landmark<N> Landmark<N>::operator+(Point2D const& rhs) const
{
    Landmark<N> result = *this;
    for(unsigned int i = 0; i < N; ++i)
    {
        result.point[i] += rhs;
    }
    return result;
}

template<unsigned int N>
Landmark<N>& Landmark<N>::operator+=(Point2D const& rhs)
{
    for(unsigned int i = 0; i < N; ++i)
    {
        this->point[i] += rhs;
    }

    return *this;
}

template<unsigned int N>
Landmark<N> Landmark<N>::operator-(Point2D const& rhs) const
{
    Landmark<N> result = *this;
    for(unsigned int i = 0; i < N; ++i)
    {
        result.point[i] -= rhs;
    }
    return result;
}

template<unsigned int N>
Landmark<N>& Landmark<N>::operator-=(Point2D const& rhs)
{
    for(unsigned int i = 0; i < N; ++i)
    {
        this->point[i] -= rhs;
    }

    return *this;
}

template<unsigned int N>
Landmark<N> Landmark<N>::operator*(Point2D const& rhs) const
{
    Landmark<N> result = *this;
    for(unsigned int i = 0; i < N; ++i)
    {
        result.point[i] *= rhs;
    }
    return result;
}

template<unsigned int N>
Landmark<N>& Landmark<N>::operator*=(Point2D const& rhs)
{
    for(unsigned int i = 0; i < N; ++i)
    {
        this->point[i] *= rhs;
    }

    return *this;
}

template<unsigned int N>
Landmark<N> Landmark<N>::operator/(Point2D const& rhs) const
{
    Landmark<N> result = *this;
    for(unsigned int i = 0; i < N; ++i)
    {
        result.point[i] /= rhs;
    }
    return result;
}

template<unsigned int N>
Landmark<N>& Landmark<N>::operator/=(Point2D const& rhs)
{
    for(unsigned int i = 0; i < N; ++i)
    {
        this->point[i] /= rhs;
    }

    return *this;
}