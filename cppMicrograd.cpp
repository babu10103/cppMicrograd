#include <iostream>
#include <stdexcept>
#include <cmath>

class Value
{
    float data;

public:
    constexpr Value(float val) : data(val) {}

    float item() const
    {
        return data;
    }

    void setData(float val) { data = val; }

    template <typename T, typename Op>
    Value doOperation(const T &other, Op operation) const
    {
        if constexpr (std::is_arithmetic<T>::value)
        {
            return Value(operation(data, other));
        }
        else if constexpr (std::is_same<T, Value>::value)
        {
            return Value(operation(data, other.data));
        }
        else
        {
            static_assert(!std::is_same<T, T>::value, "Unsupported type for operation");
        }
    }

    template <typename T>
    Value operator+(const T &other) const
    {
        return doOperation(other, std::plus<>());
    }
    template <typename T>
    Value operator-(const T &other) const
    {
        return doOperation(other, std::minus<>());
    }

    template <typename T>
    Value operator*(const T &other) const
    {
        return doOperation(other, std::multiplies<>());
    }

    template <typename T>
    Value operator/(const T &other) const
    {
        if constexpr (std::is_arithmetic<T>::value)
        {
            if (other == 0)
            {
                throw std::runtime_error("Division by zero");
            }
            return Value(data / other);
        }
        else if constexpr (std::is_same<T, Value>::value)
        {
            if (other.data == 0)
            {
                throw std::runtime_error("Division by zero");
            }
            return Value(data / other.data);
        }
        else
        {
            static_assert(!std::is_same<T, T>::value, "Unsupported type for operator /");
        }
    }

    template <typename T>
    Value operator^(const T& other) const {
        if constexpr (std::is_arithmetic<T>::value) {
            return Value(std::pow(data, other));
        } else if constexpr (std::is_same<T, Value>::value) {
            return Value(std::pow(data, other.data));
        } else {
            static_assert(!std::is_same<T, T>::value, "Unsupported type for operator ^");
        }
    }

    Value exp() const {
        return Value(std::exp(data));
    }

    friend std::ostream &operator<<(std::ostream &stream, const Value &value);
};

std::ostream &operator<<(std::ostream &stream, const Value &value)
{
    stream << "Value(" << value.data << ")";
    return stream;
}

int main()
{
    Value a(19.3f), b(34.555f);
    Value c = a + b;
    Value d = c / 2.4;
    Value e = d * 2;
    std::cout << d << ", " << e << std::endl;
}