#ifndef FCANN_H_INCLUDED
#define FCANN_H_INCLUDED

#include<string>
#include<map>
#include<math.h>
#include "Matrix.h"

/*
FCANN: Fully Connected Artificial Neural Network
*/

/**
template<typename T>
class Activation
{
public:
    T operator()()
};
**/

template<typename T>
class FCANN //: public Matrix<T>
{
public:
    // default constructor: yields I/O function
    FCANN(unsigned inputs, unsigned outputs, unsigned batch_size, double lr);
    FCANN(unsigned inputs, unsigned outputs, unsigned batch_size, double lr, \
          std::vector<unsigned>& architectureVect);
    ~FCANN();
    void setLearningRate(const T& learning_rate);

    Matrix<T> forwardPropagation(Matrix<T>& input);

    /// Cost functions
    // mean squared error
    Matrix<T> MSE(Matrix<T>& prediction, Matrix<T>& labels);
    // mean squared error derivative
    Matrix<T> MSE_der(Matrix<T>& prediction, Matrix<T>& labels);
    // cross-entropy loss
    Matrix<T> binaryCrossEntropy(Matrix<T>& prediction, Matrix<T>& labels);
    // cross-entropy loss derivative
    Matrix<T> binaryCrossEntropy_der(Matrix<T>& prediction, Matrix<T>& labels);
    // cross-entropy loss for multi-class outputs
    Matrix<T> crossEntropy(Matrix<T>& prediction, Matrix<T>& labels);
    // cross-entropy loss derivative for multi-class outputs
    Matrix<T> crossEntropy_der(Matrix<T>& prediction, Matrix<T>& labels);

    void train(Matrix<T>& _inputs, Matrix<T>& _labels, unsigned epochs, double eps_t=-1);
    void backwardPropagation(Matrix<T>& prediction, Matrix<T>& labels);
    /// gradient descent
    void computeGradients(Matrix<T>& prediction, Matrix<T>& labels); /// not implemented yet
    /// optimizers
    void SGD(Matrix<T>& prediction, Matrix<T>& labels); /// not implemented yet
    void momentum(Matrix<T>& prediction, Matrix<T>& labels, const T& alpha=0.9);



    void resetCache(); /// not implemented yet

    void printWeights() const;
    void printActivations() const; /// !!! bug with indices !!!
    void printGradients() const;
    void printErrors() const;

private:
    unsigned m_inputs;
    unsigned m_outputs;
    unsigned m_batch_size;
    double m_lr;

    // ANN architecture // size must be equal to mat_number - 1 if values are assigned to it
    std::vector<unsigned> m_architectureVect;
    // [(a0, b0), ..., (an, bn)]
    std::vector<std::pair<unsigned, unsigned> > m_dimensionsVect;
    // map that contains weight matrices as values
    std::map<std::string, Matrix<T>* > m_weightsMap;
    // map that contains activated layers as values.
    //keys for weights and activated layers are the same
    // such that m_weights.keys == m_activatedLayersMap
    std::map<std::string, Matrix<T>* > m_activationsMap;
    /** store gradients: DeltaW, deltaA^(l+1) **/
    std::map<std::string, Matrix<T>* > m_gradientsMap;
    std::map<std::string, Matrix<T>* > m_errorsMap;
    /// optimization algorithms arguments
    std::map<std::string, Matrix<T>* > m_velocitiesMap;
};
#endif // FCANN_H_INCLUDED
