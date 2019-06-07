#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <iostream>
#include <vector>
#include <string>
#include <sstream> // convert double to string
#include <cstdlib>
//#include <random> // rng
#include <ctime>
#include <math.h>

/// included for rng seed
#include <ctime>
/// get seed etc for random
#include <random> /// std::default_random_engine
#include <chrono> /// std::chrono::system_clock

#include <algorithm> /// std::random_shuffle(..., ...)
#include <iterator> /// class Vector2D_iterator ...

// template static variables
template <typename T>
struct VARIABLES
{
    static unsigned COUNTER;
};

// an iterator over a vector of vectors
// https://stackoverflow.com/questions/1784573/iterator-for-2d-vector
template<typename T>
class Vector2D_iterator : public std::iterator<std::bidirectional_iterator_tag, T>
{
public:
    static Vector2D_iterator<T> begin(std::vector<std::vector<T> >& vect){
        return Vector2D_iterator(&vect, 0, 0);
    }
    static Vector2D_iterator<T> end(std::vector<std::vector<T> >& vect){
        return Vector2D_iterator(&vect, vect.size(), 0);
    }

    Vector2D_iterator() = default;
    // ++prefix operator
    Vector2D_iterator& operator++()
    {
        // if we havent reached the end of this sub-vector
        if(idxInner+1 < (*vect)[idxOuter].size())
        {
            // go to next element
            ++idxInner;
        }
        else
        {
            // otherwise skip to the next sub-vector, and keep skipping over empty
            // ones until we reach a non-empty one or the end
            do
            {
                ++idxOuter;
            } while(idxOuter < (*vect).size() && (*vect)[idxOuter].empty());

            // go to the beginning of this vector
            idxInner = 0;
        }
        return *this;
    }
    // --prefix operator
    Vector2D_iterator& operator--()
    {
        // if we have reached the start of this sub-vector
        if(idxInner > 0)
        {
            // go to the previous element
            --idxInner;
        }
        else
        {
            // otherwise skip to the previous sub-vector, and keep skipping over empty
            // ones until we reach a non-empty one
            do
            {
                --idxOuter;
            } while ((*vect)[idxOuter].empty());
            // go to the end of this vector
            idxInner = (*vect)[idxOuter].size() - 1;
        }
        return *this;
    }
    // postfix++ operator
    Vector2D_iterator operator++(int)
    {
        T retval = *this;
        ++(*this);
        return retval;
    }
    // postfix-- operator
    Vector2D_iterator operator--(int)
    {
        T retval = *this;
        --(*this);
        return retval;
    }
    bool operator==(const Vector2D_iterator& other) const
    {
        return other.vect == vect && other.idxOuter == idxOuter && other.idxInner == idxInner;
    }
    bool operator!=(const Vector2D_iterator& other) const
    {
        return !(*this == other);
    }
    const T& operator*() const
    {
        return *this;
    }
    T& operator*()
    {
        return (*vect)[idxOuter][idxInner];
    }
    const T& operator->() const
    {
        return *this;
    }
    T& operator->()
    {
        return *this;
    }

private:
    Vector2D_iterator(std::vector<std::vector<T> >* _vect,
                      std::size_t _idxOuter,
                      std::size_t _idxInner) : vect(_vect), idxOuter(_idxOuter), idxInner(_idxInner) {}
    std::vector<std::vector<T> >* vect = nullptr;
    std::size_t idxOuter = 0;
    std::size_t idxInner = 0;

};

//class pre-declaration for friend operator<< <>(...)
template<typename T>
class Matrix;

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix);

template<typename T>
class Matrix
{
public:
    Matrix();
    Matrix(unsigned rows, unsigned columns, const T& initValue);

    /// not (completely verified)
    Matrix(unsigned rows, unsigned columns, std::string distribution="gaussian", \
           double mean=0, double sigma=1, \
           double inf=-1, double sup=1);
    Matrix(const Matrix<T>& other);
    virtual ~Matrix();

    //instanciation speciation matrices
    void eye();
    Matrix<T> getSubmatrix(unsigned from_i, unsigned from_j, \
                           unsigned to_i,   unsigned to_j);
    void insertFrame(Matrix<T>& other,\
                    unsigned from_i=0, unsigned from_j=0);

    Matrix<T> hStack(const Matrix<T>& other);
    Matrix<T> vStack(const Matrix<T>& other);
    Matrix<T> hSum();
    Matrix<T> vSum();
    Matrix<T>& vShuffle();

    Matrix<T>& activationFunction(const std::string& activation_func);
    Matrix<T>& deriveActivation(const std::string& activation_func); /// not implemented
    /// if we do not want to modify the matrix directly -> no reference
    Matrix<T> getActivationDerivative(const std::string& activation_func);

    // not operator based operators
    Matrix<T>& elemProduct(const Matrix<T>& other); /// needs error management
    // horizontal product broadcasting
    Matrix<T>& hProdBroadcast(const Matrix<T>& other);
    // vertical product broadcasting
    Matrix<T>& vProdBroadcast(const Matrix<T>& other);

    //Overloading operators for mathematical purposes
    Matrix<T>& operator=(const Matrix<T>& other);

    //Matrix - matrix operations
    Matrix<T> operator+(const Matrix<T>& other);
    Matrix<T>& operator+=(const Matrix<T>& other);
    Matrix<T> operator-(const Matrix<T>& other);
    Matrix<T>& operator-=(const Matrix<T>& other);
    Matrix<T> operator*(const Matrix<T>& other);
    Matrix<T>& operator*=(const Matrix<T>& other);
    Matrix<T> operator/(const Matrix<T>& other);
    Matrix<T>& operator/=(const Matrix<T>& other);
    Matrix<T> transpose();
    //transforms the object itself
    Matrix<T>& self_transpose();

    //Matrix - vector operations
    std::vector<T> operator*(const std::vector<T>& other);
    std::vector<T> getDiagonalVect();
    Matrix<T> getDiagonal();


    //Access individual elements
    T& operator()(const unsigned& row, const unsigned& column);
    const T& operator()(const unsigned& row, const unsigned& column) const;

    //Access row - column sizes
    unsigned getRows() const;
    unsigned getCols() const;

    //Matrix - scalar operations
    Matrix<T> operator+(const T& scalar);
    Matrix<T> operator-(const T& scalar);
    Matrix<T> operator*(const T& scalar);
    Matrix<T> operator/(const T& scalar);
    // Matrix - scalar operations (right hand side)
    Matrix<T>& operator+=(const T& scalar);
    Matrix<T>& operator*=(const T& scalar);
    Matrix<T>& operator-=(const T& scalar); ///error here: should be m_matrix[i][j] - scalar
    Matrix<T>& operator/=(const T& scalar); ///error here: should be m_matrix[i][j] / scalar

    // visualization methods
    //stream based visualization
    void print(std::ostream& STREAM) const;
    //direct object visualization method
    void print() const;

private:
    std::vector<std::vector<T> > m_matrix;
    unsigned m_rows;
    unsigned m_columns;

    // friend function of a template class
    /* http://www.cplusplus.com/forum/general/45776/ */
    /* friend with templates explanation
    https://www.ibm.com/support/knowledgecenter/SSGH3R_16.1.0/com.ibm.xlcpp161.aix.doc/language_ref/friends_and_templates.html
    */
    friend std::ostream& operator<< <>(std::ostream& out, const Matrix<T>& matrix);
};

/// START - NON MEMBER FUNCTIONS
/**
// scalar - Matrix addition
template<typename T, typename T2>
Matrix<T> operator+(const T2& scalar, Matrix<T> rhs)
{
    return rhs += static_cast<T>(scalar);
}

// scalar - Matrix multiplication
template<typename T, typename T2>
Matrix<T> operator*(const T2& scalar, Matrix<T> rhs)
{
    return rhs *= static_cast<T>(scalar);
}

// scalar - Matrix subtraction
template<typename T, typename T2>
Matrix<T> operator-(const T2& scalar, Matrix<T> rhs)
{
    return rhs -= static_cast<T>(scalar);
}

// scalar - Matrix division
template<typename T, typename T2>
Matrix<T> operator/(const T2& scalar, Matrix<T> rhs)
{
    return rhs /= static_cast<T>(scalar);
}

// opposite matrix E.g: new_matrix = - matrix;
template<typename T>
Matrix<T> operator-(Matrix<T> rhs)
{
    return rhs *= (-1);
}
**/

// scalar - Matrix addition
template<typename T>
Matrix<T> operator+(const double& scalar, Matrix<T> rhs)
{
    return rhs += scalar;
}

// scalar - Matrix multiplication
template<typename T>
Matrix<T> operator*(const double& scalar, Matrix<T> rhs)
{
    return rhs *= scalar;
}

// scalar - Matrix subtraction
template<typename T>
Matrix<T> operator-(const double& scalar, Matrix<T> rhs)
{
    return rhs -= scalar;
}

// scalar - Matrix division
template<typename T>
Matrix<T> operator/(const double& scalar, Matrix<T> rhs)
{
    return rhs /= scalar;
}

// opposite matrix E.g: new_matrix = - matrix;
template<typename T>
Matrix<T> operator-(Matrix<T> rhs)
{
    return rhs *= (-1);
}

/// END   - NON MEMBER FUNCTIONS

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& matrix){
    matrix.print(out);
    return out;
}

template<typename T>
class Activation
{
public:
    Activation(const std::string& activation) : m_activation{activation}
    {}

    Activation(const std::string& activation, \
               const Matrix<T>& sum_exp_layer, \
               unsigned m, unsigned n) : m_activation{activation}, \
                                         m_m{m}, m_n{n}
    {
        /** m and n are the dimensions of the
            layer for the softmax activation **/
        m_i = 0; m_j = 0; /// current indices, incremented inside softmax condition
        m_sum_exp_layer = sum_exp_layer;
    }

    void operator()(T& element){
        if(m_activation == "sigmoid")
            element = 1 / (1 + exp(-element));
        else
            if(m_activation == "relu"){
                if(element <= 1e-6)
                    element = 0;
            }
        else
            if(m_activation == "softmax"){
                element = exp(element) / m_sum_exp_layer(m_i, 0);
                m_j += 1;
                if(m_j == m_n){m_j = 0; m_i++;}
                if(m_i == m_m) m_i = 0;
            }
        else // this is NOT an ACTIVATION function
            if(m_activation == "log"){
                //if(element < 0.0001)
                //    element = 9999;
                //else
                    element = log(element+1e-6);
            }
        else // this is NOT an ACTIVATION function
            if(m_activation == "exp"){
                element = exp(element);
            }
        else // this is NOT an ACTIVATION function
            if(m_activation == "clip"){
                if(element > 1)
                    element = 1; /// deactivated clipping
                else
                    if(element < -1)
                        element = -1; //-10;
            }
    }

private:
    std::string m_activation;
    /// summed, column wise exponentiated layer, for the softmax activation
    Matrix<T> m_sum_exp_layer;
    unsigned m_i; unsigned m_j;
    unsigned m_m; unsigned m_n;
};


template<typename T>
class ActivationDerivative
{
public:
    ActivationDerivative(const std::string& activation) : m_activation{activation}
    {}

    void operator()(T& element){
        if(m_activation == "sigmoid")
            element = element * (1 - element);//1 / (1 + exp(-element));
        else
            if(m_activation == "relu"){
                if(element <= 0)
                    element = 0;
                else
                    element = 1;
            }
        else // this is NOT an ACTIVATION function derivative
            if(m_activation == "log")
                element = 1 / element;
    }
private:
    std::string m_activation;
};

#endif
