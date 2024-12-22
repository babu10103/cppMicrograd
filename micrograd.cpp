#include <iostream>
#include <stdexcept>
#include <set>
#include <cmath>
#include <functional>
#include <random>
class Value
{
    std::string _op;
    std::set<Value *> _prev;
    std::function<void()> _backward;

public:
    double data;
    double grad;
    std::string label;

    Value(double data, const std::string &_op = "", const std::string &label = "", const std::set<Value *> &_children = {})
        : data(data), _op(_op), label(label), grad(0.0), _prev(_children), _backward([]() {}) {}
    // Default constructor (initializes data and grad to 0)
    Value() : data(0.0), grad(0.0) {}  // Default constructor

    double item() const
    {
        return this->data;
    }

    void setData(double val)
    {
        this->data = val;
    }

    Value operator+(Value &other)
    {
        Value out(this->data + other.data, "+", "", {this, &other});

        out._backward = [this, &other, &out]()
        {
            this->grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };

        return out;
    }
    Value operator+(double other) const
    {
        Value result(this->data + other);
        return result;
    }

    friend Value operator+(double lhs, const Value &rhs)
    {
        Value result(0.0);
        result.data = lhs + rhs.data;
        return result;
    }

    Value operator*(Value &other)
    {
        Value out(this->data * other.data, "*", "", {this, &other});

        out._backward = [this, &other, &out]()
        {
            this->grad += other.data * out.grad;
            other.grad += this->data * out.grad;
        };
        return out;
    }
    Value operator*(double scalar)
    {
        Value result(scalar);
        return *this * result;
    }

    friend Value operator*(double lhs, Value &rhs)
    {
        Value result(0.0);
        result.data = lhs * rhs.data;
        return result;
    }

    Value operator/(Value &other)
    {
        return *this * (1.0 / other.data);
    }
    Value operator/(double &scalar)
    {
        Value other(scalar, "", "", {});
        return *this / other;
    }
    friend Value operator/(double scalar, Value &val)
    {
        return val * (1.0 / scalar);
    }
    Value operator^(double exponent)
    {
        std::string op = "^" + std::to_string(exponent);
        Value out(std::pow(this->data, exponent), op, "", {this});
        // Backward function for gradient computation
        out._backward = [this, &exponent, &out]()
        {
            // Gradients for a^b:
            // da/dx = b * a^(b-1)
            // db/dx = a^b * log(a)
            this->grad += exponent * std::pow(this->data, exponent - 1) * out.grad;
            exponent += std::pow(this->data, exponent) * std::log(this->data) * out.grad;
        };
        return out;
    }

    Value tanh()
    {
        // Compute tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        double data_tanh = (std::exp(2 * data) - 1) / (std::exp(2 * data) + 1);
        Value out(data_tanh, "tanh(" + std::to_string(data) + ")", "", {this});
        out._backward = [this, data_tanh, &out]()
        {
            this->grad += (1 - data_tanh * data_tanh) * out.grad;
        };

        return out;
    }

    Value exp()
    {
        double data_exp = std::exp(data); // e^x
        Value out(data_exp, "exp(" + std::to_string(data) + ")", "", {this});
        out._backward = [this, data_exp, out]()
        {
            this->grad += data_exp * out.grad;
        };

        return out;
    }

    void backward()
    {
        std::vector<Value *> topo_order;
        std::set<Value *> visited;
        std::function<void(Value *)> build_topo_order = [&](Value *v)
        {
            if (visited.find(v) != visited.end())
            {
                return;
            }
            visited.insert(v);
            for (Value *child : v->_prev)
            {
                build_topo_order(child);
            }
            topo_order.push_back(v);
        };
        build_topo_order(this);
        reverse(topo_order.begin(), topo_order.end());
        for (Value *v : topo_order)
        {
            std::cout << v->data << ", " << v->grad << std::endl;
            v->_backward();
        }
    }

    friend std::ostream &operator<<(std::ostream &stream, const Value &value);
};

namespace util {
    // Modify the function to accept the range as a const lvalue reference
    double getRandomValueInRange(const std::pair<double, double>& range) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(range.first, range.second);
        return dis(gen);
    }

    // Function to create random values
    std::vector<Value> createRandomValues(int nin) {
        std::vector<Value> values;
        std::pair<double, double> range = {-1.0, 1.0};
        for (int i = 0; i < nin; ++i) {
            values.push_back(Value(getRandomValueInRange(range)));  // Passing range as lvalue
        }
        return values;
    }
}

class Neuron {
public:
    int nin;
    std::vector<Value> weights;
    Value bias;

    Neuron(int nin) : nin(nin), bias(Value(util::getRandomValueInRange({-1.0, 1.0}))) {
        weights = util::createRandomValues(nin);
    }

    Value operator()(const std::vector<Value>& x) {
        double weighted_sum = std::accumulate(weights.begin(), weights.end(), bias.data, 
            [&x, index = 0] (double sum, const Value& wi) mutable {
            return sum + (wi.data  * x[index++].data);
        });
        return Value(std::tanh(weighted_sum));
    }

    std::vector<Value> parameters() {
        std::vector<Value> params = weights;
        params.push_back(bias);
        return params;
    }

};


std::ostream &operator<<(std::ostream &stream, const Value &value)
{
    stream << "Value(" << value.data << ")";
    return stream;
}

int main()
{
    Value a = Value(2.0);
    a.label = "a";
    Value b = Value(0.0);
    b.label = "b";
    Value c = Value(-3.0);
    c.label = "c";
    Value d = Value(1.0);
    d.label = "d";
    Value e = a * c;
    e.label = "a*c";
    Value f = b * d;
    f.label = "b*d";
    Value g = e + f;
    g.label = "g";
    Value bias = Value(6.8813735870195432);
    bias.label = "bias";
    Value n = g + bias;
    n.label = "n";
    Value o = n.tanh();
    o.label = "o";

    std::cout << "Backward pass" << std::endl;
    o.grad = 1.0;
    o.backward();

    std::cout << "Gradients" << std::endl;
    std::cout << a.grad <<", " + a.label << std::endl;
    std::cout << b.grad << ", " + b.label << std::endl;
    std::cout << c.grad << ", " + c.label << std::endl;
    std::cout << d.grad << ", " + d.label << std::endl;
    std::cout << e.grad << ", " + e.label << std::endl;
    std::cout << f.grad << ", " + f.label << std::endl;
    std::cout << g.grad << ", " + g.label << std::endl;
    std::cout << bias.grad << ", " + bias.label << std::endl;
    std::cout << n.grad << ", " + n.label << std::endl;
    std::cout << o.grad << ", " + o.label << std::endl;
    return 0;
}