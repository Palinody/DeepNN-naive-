#include "FCANN.h"

template class FCANN<float>;
template class FCANN<double>;

template<typename T>
FCANN<T>::FCANN(unsigned inputs, unsigned outputs, unsigned batch_size, double lr): m_inputs{inputs}, \
                                                                 m_outputs{outputs}, \
                                                                 m_batch_size{batch_size}, \
                                                                 m_lr{lr}
{
    /**
    The constructor may seem a little bit complicated for such a simple network,
    but I want every instance of the class to hold the same arguments
    **/
    // bias term -> +1
    m_architectureVect.push_back(m_inputs);
    std::pair<unsigned, unsigned> myPair(0, 0);
    myPair.first = m_architectureVect[m_architectureVect.size()-1];
    myPair.second = m_outputs;
    m_architectureVect.push_back(myPair.second);
    m_dimensionsVect.push_back(myPair);

    /// input place holder -> m_activationsMap["a0"]
    m_activationsMap.insert(std::make_pair("a"+std::to_string(0), \
                                           new Matrix<T>(m_batch_size, \
                                                         m_inputs+1, 0)));

    m_weightsMap.insert(std::make_pair("w"+std::to_string(1), \
                                    new Matrix<T>(m_architectureVect[0]+1, \
                                                  m_architectureVect[1], "gaussian", 0, 1)));
    m_activationsMap.insert(std::make_pair("a"+std::to_string(1), \
                                           new Matrix<T>(m_batch_size, \
                                                         m_architectureVect[1]+1, 0)));
    // allocating and initialization of gradient and error mapping
    m_gradientsMap.insert(std::make_pair("D"+std::to_string(1), \
                                         new Matrix<T>(m_architectureVect[0]+1, \
                                                       m_architectureVect[1], 0)));
    // error matrix don't include the bias -> columns("d"+i) = columns("a"+i) - 1
    m_errorsMap.insert(std::make_pair("d"+std::to_string(1), \
                                           new Matrix<T>(m_batch_size, \
                                                         m_architectureVect[1], 0)));

}

template<typename T>
FCANN<T>::FCANN(unsigned inputs, unsigned outputs, unsigned batch_size, double lr, \
                std::vector<unsigned>& architectureVect): m_inputs{inputs}, \
                                                          m_outputs{outputs}, \
                                                          m_batch_size{batch_size}, \
                                                          m_lr{lr}
{
    /// DIMENSIONS VECTOR INITIALIZATION
    m_architectureVect.push_back(m_inputs);
    m_architectureVect.insert(m_architectureVect.end(), architectureVect.begin(), architectureVect.end());
    m_architectureVect.push_back(m_outputs);

    unsigned depth{m_architectureVect.size()};
    for(unsigned i{1}; i < depth; i++)
    {
        std::pair<unsigned, unsigned> myPair(0, 0);
        myPair.first = m_architectureVect[i-1];
        myPair.second = m_architectureVect[i];
        //m_architectureVect.push_back(myPair.second);
        m_dimensionsVect.push_back(myPair);
    }

    /// input place holder -> m_activationsMap["a0"]
    m_activationsMap.insert(std::make_pair("a"+std::to_string(0), \
                                           new Matrix<T>(m_batch_size, \
                                                         m_inputs+1, 0)));

    for(unsigned i{0}; i < m_dimensionsVect.size(); i++)
    {
        m_weightsMap.insert(std::make_pair("w"+std::to_string(i+1), \
                                           new Matrix<T>(m_dimensionsVect[i].first+1, \
                                                         m_dimensionsVect[i].second, "Xavier", 0, 1)));

        m_activationsMap.insert(std::make_pair("a"+std::to_string(i+1), \
                                           new Matrix<T>(m_batch_size, \
                                                         m_dimensionsVect[i].second+1, 1)));
        // allocating and initialization of gradient and error mapping
        m_gradientsMap.insert(std::make_pair("D"+std::to_string(i+1), \
                                             new Matrix<T>(m_dimensionsVect[i].first+1, \
                                                           m_dimensionsVect[i].second, 0)));
        // error matrix don't include the bias -> columns("d"+i) = columns("a"+i) - 1
        m_errorsMap.insert(std::make_pair("d"+std::to_string(i+1), \
                                           new Matrix<T>(m_batch_size, \
                                                         m_dimensionsVect[i].second, 0)));
        // initialize velocities for momentum optimization algorithm
        m_velocitiesMap.insert(std::make_pair("v"+std::to_string(i+1), \
                                              new Matrix<T>(m_dimensionsVect[i].first+1, \
                                                            m_dimensionsVect[i].second, 0)));
    }
}

template<typename T>
FCANN<T>::~FCANN(){
    delete m_activationsMap["a"+std::to_string(0)];
    for(unsigned i{0}; i < m_dimensionsVect.size(); i++){
        delete m_weightsMap["w"+std::to_string(i+1)];
        delete m_activationsMap["a"+std::to_string(i+1)];
        delete m_gradientsMap["D"+std::to_string(i+1)];
        delete m_errorsMap["d"+std::to_string(i+1)];
        delete m_velocitiesMap["v"+std::to_string(i+1)];
    }
}

template<typename T>
void FCANN<T>::setLearningRate(const T& learning_rate){m_lr = learning_rate;}

template<typename T>
Matrix<T> FCANN<T>::forwardPropagation(Matrix<T>& input){
    // adding bias to current layer (input layer)
    Matrix<T> bias_vector(m_batch_size, 1, 1);
    Matrix<T> current_layer = input.hStack(bias_vector);
    /// replace the place holder by real input
    /// input placeholder -> m_activationsMap["a0"]
    *m_activationsMap["a"+std::to_string(0)] = current_layer;

    for(unsigned i{1}; i < m_architectureVect.size(); i++){
        /// activations last col. have bias terms -> need to place subframe that is
        /// the result of the dot product btw curr_layer and curr_weights
        current_layer *= *m_weightsMap["w"+std::to_string(i)];
        if(i == m_architectureVect.size()-1)
            current_layer.activationFunction("softmax"); /// output activation
        else
            current_layer.activationFunction("relu");
        m_activationsMap["a"+std::to_string(i)]->insertFrame(current_layer);

        current_layer = *m_activationsMap["a"+std::to_string(i)];
    }
    /// return prediction without bias column
    return current_layer.getSubmatrix(0, 0, current_layer.getRows(), current_layer.getCols()-1);
}

template<typename T>
void FCANN<T>::backwardPropagation(Matrix<T>& prediction, Matrix<T>& labels){
    Matrix<T> prev_layer;
    Matrix<T> curr_layer;
    Matrix<T> next_layer;
    Matrix<T> curr_delta;
    Matrix<T> next_delta;
    Matrix<T> Grad;
    //curr_delta = MSE_der(prediction, labels);
    curr_delta = binaryCrossEntropy_der(prediction, labels);
    curr_delta.elemProduct(prediction.getActivationDerivative("sigmoid"));
    //curr_delta = crossEntropy_der(prediction, labels);

    for(unsigned l{m_architectureVect.size()-1}; l >= 1; l--){
        if(l < m_architectureVect.size()-1){
            curr_layer = *m_activationsMap["a"+std::to_string(l)];
            unsigned curr_layer_rows = curr_layer.getRows();
            unsigned curr_layer_cols = curr_layer.getCols();
            /// !!! remember that delta IS NEXT DELTA at this state !!! -> next_delta = curr_delta
            next_delta = curr_delta;
            unsigned rows = m_weightsMap["w"+std::to_string(l+1)]->getRows();
            unsigned cols = m_weightsMap["w"+std::to_string(l+1)]->getCols();
            Matrix<T> tempWeights = *m_weightsMap["w"+std::to_string(l+1)];
            tempWeights = tempWeights.getSubmatrix(0, 0, rows-1, cols).transpose();
            curr_delta = next_delta * tempWeights;

            curr_delta.elemProduct(curr_layer.getSubmatrix(0, 0, curr_layer_rows, curr_layer_cols-1).getActivationDerivative("relu"));
        }

        /// store delta
        *m_errorsMap["d"+std::to_string(l)] = curr_delta;

        prev_layer = *m_activationsMap["a"+std::to_string(l-1)];
        Grad = (prev_layer.transpose() * curr_delta) / m_batch_size;
        Grad.activationFunction("clip"); /// applied gradient clipping
        /// store gradients
        *m_gradientsMap["D"+std::to_string(l)] = Grad;
        /// Momentum: v = alpha * v - m_lr * Grad
        ///*m_weightsMap["w"+std::to_string(l)] -= (Grad * m_lr); //W += -alpha*Grad
    }
    for(unsigned l{m_architectureVect.size()-1}; l >= 1; l--){
        /// Momentum: v = alpha * v - m_lr * Grad
        Grad = *m_gradientsMap["D"+std::to_string(l)];
        *m_weightsMap["w"+std::to_string(l)] -= (Grad * m_lr); //W += -alpha*Grad
    }
}

template<typename T>
void FCANN<T>::momentum(Matrix<T>& prediction, Matrix<T>& labels, const T& alpha)
{
    /**
    alpha:
        momentum parameter: generally set to 0.5, 0.9 or 0.99
    optimization idea for initializing velocity:
        check dimensions of the weights and set v to be Matrix<T> v(rows.max, cols.max, 0)
        then pick a sub matrix from this matrix and set it to curr_v. Then compute...
    **/
    Matrix<T> prev_layer;
    Matrix<T> curr_layer;
    Matrix<T> next_layer;
    Matrix<T> curr_delta;
    Matrix<T> next_delta;
    Matrix<T> Grad;
    ///curr_delta = (prediction - labels) / m_batch_size;
    //curr_delta = MSE_der(prediction, labels);
    //curr_delta = binaryCrossEntropy_der(prediction, labels);
    //curr_delta.elemProduct(prediction.getActivationDerivative("sigmoid"));
    curr_delta = crossEntropy_der(prediction, labels);

    for(unsigned l{m_architectureVect.size()-1}; l >= 1; l--){
        if(l < m_architectureVect.size()-1){
            curr_layer = *m_activationsMap["a"+std::to_string(l)];
            unsigned curr_layer_rows = curr_layer.getRows();
            unsigned curr_layer_cols = curr_layer.getCols();
            /// !!! remember that delta IS NEXT DELTA at this state !!! -> next_delta = curr_delta
            next_delta = curr_delta;
            unsigned rows = m_weightsMap["w"+std::to_string(l+1)]->getRows();
            unsigned cols = m_weightsMap["w"+std::to_string(l+1)]->getCols();
            Matrix<T> tempWeights = *m_weightsMap["w"+std::to_string(l+1)];
            tempWeights = tempWeights.getSubmatrix(0, 0, rows-1, cols).transpose();
            curr_delta = next_delta * tempWeights;

            curr_delta.elemProduct(curr_layer.getSubmatrix(0, 0, curr_layer_rows, curr_layer_cols-1).getActivationDerivative("relu"));
        }

        /// store delta
        *m_errorsMap["d"+std::to_string(l)] = curr_delta;

        prev_layer = *m_activationsMap["a"+std::to_string(l-1)];
        /// start: adding weight decay
        double reg_factor = 0.05; /// added
        unsigned rows = m_weightsMap["w"+std::to_string(l)]->getRows()-1; // we dont want to update the bias
        unsigned cols = m_weightsMap["w"+std::to_string(l)]->getCols(); /// added
        Matrix<T> weight_decay = reg_factor * m_weightsMap["w"+std::to_string(l)]->getSubmatrix(0, 0, rows, cols);
        /// end: adding weight decay
        Grad = (prev_layer.transpose() * curr_delta) / m_batch_size;
        Matrix<T> TEMP = Grad.getSubmatrix(0, 0, rows, cols) + weight_decay / m_batch_size; /// added
        Grad.insertFrame(TEMP, 0, 0); ///with weight decay
        Grad.activationFunction("clip"); /// applied gradient clipping
        /// store gradients
        *m_gradientsMap["D"+std::to_string(l)] = Grad;
        /// Momentum: v = alpha * v - m_lr * Grad
        //*m_velocitiesMap["v"+std::to_string(l)] = (*m_velocitiesMap["v"+std::to_string(l)] * alpha) - (Grad * m_lr);

        ///*m_weightsMap["w"+std::to_string(l)] += *m_velocitiesMap["v"+std::to_string(l)];
    }
    for(unsigned l{m_architectureVect.size()-1}; l >= 1; l--){
        Grad = *m_gradientsMap["D"+std::to_string(l)];
        /// Momentum: v = alpha * v - m_lr * Grad
        *m_velocitiesMap["v"+std::to_string(l)] = (*m_velocitiesMap["v"+std::to_string(l)] * alpha) - (Grad * m_lr);

        Grad = *m_velocitiesMap["v"+std::to_string(l)];
        *m_weightsMap["w"+std::to_string(l)] += Grad; //W += -alpha*Grad
    }
}

template<typename T>
Matrix<T> FCANN<T>::MSE(Matrix<T>& prediction, Matrix<T>& labels){
    Matrix<T> error_matrix = prediction - labels;
    error_matrix = error_matrix.transpose() * error_matrix;
    error_matrix = error_matrix.getDiagonal() / (2*m_batch_size); // horizontal sum?
    return error_matrix;
}

template<typename T>
Matrix<T> FCANN<T>::MSE_der(Matrix<T>& prediction, Matrix<T>& labels){
    ///return error_der_matrix.vSum() / (m_batch_size);
    return prediction - labels;
}

// binary cross entropy for output layer with 1 prediction: 1 or 0
template<typename T>
Matrix<T> FCANN<T>::binaryCrossEntropy(Matrix<T>& prediction, Matrix<T>& labels){
    /// -{(1 - labels) * log(1 - prediction) + labels * (prediction)}
    Matrix<T> p = prediction;
    Matrix<T> q = 1 - prediction;
    Matrix<T> p_label = labels;
    Matrix<T> q_label = 1 - labels;
    //Matrix<T> error_matrix = ((1 - labels).elemProduct((1 - prediction).activationFunction("log"))\
                             + labels.elemProduct(prediction.activationFunction("log")) );
    Matrix<T> error_matrix = - (q_label.elemProduct(q.activationFunction("log")) \
                                + p_label.elemProduct(p.activationFunction("log")));
    /// method 1
    error_matrix = error_matrix.vSum() / m_batch_size;
    /// method 2
    //error_matrix = error_matrix.transpose() * error_matrix;
    //error_matrix = error_matrix.getDiagonal() / m_batch_size;
    return error_matrix;
}

// binary cross entropy derivative for output layer with 1 prediction: 1 or 0
template<typename T>
Matrix<T> FCANN<T>::binaryCrossEntropy_der(Matrix<T>& prediction, Matrix<T>& labels){
    /// https://deepnotes.io/softmax-crossentropy
    /// https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

    /// -> loss function
    /// -{(1 - labels) * log(1 - prediction) + labels * (prediction)}
    // because of the inner derivative of the "- y_hat"
    // term in the "1/(1 - y_hat)" expression
    // -> "-[expression]"
    // the derivative of the activation is handled inside of the
    // SGD method: generally sigmoid_der for logistic regression
    /// -> loss function derivative
    /// + (1 - y)/(1 - y_hat) - (y)/(y_hat)
    return (1 - labels)/(1 - prediction) - labels/prediction;
}

template<typename T>
Matrix<T> FCANN<T>::crossEntropy(Matrix<T>& prediction, Matrix<T>& labels){
    /// I can do either element-wise product to cancel the terms that are not the desired class
    /// or get the desired class index and apply it to the corresponding prediction
    /*
    double reg_factor;
    unsigned n_layer = m_architectureVect.size() - 1;
    unsigned rows = m_weightsMap["w"+std::to_string(n_layer)]->getRows() - 1;
    unsigned cols = m_weightsMap["w"+std::to_string(n_layer)]->getCols();

    Matrix<T> W = m_weightsMap["w"+std::to_string(n_layer)]->getSubmatrix(0, 0, rows, cols);
    W = reg_factor/2 * (W * W.transpose());
    W = (W.getDiagonal()).hSum();
    */
    return - ( (labels.elemProduct( prediction.activationFunction("log")) ).vSum() ) / m_batch_size; //+ W;
}

template<typename T>
Matrix<T> FCANN<T>::crossEntropy_der(Matrix<T>& prediction, Matrix<T>& labels){
    return prediction - labels;
}

template<typename T>
void FCANN<T>::train(Matrix<T>& _inputs, Matrix<T>& _labels, \
                     unsigned epochs, \
                     double eps_t){
    /**
    0)  ** Miscellaneous **
        find a way to:
            - make the user choose the Cost function
            he wants to use.
            - let him specify at which rate he visualizes
            the current cost function ; normalize the
            "visualization curve" such that it always fits
            into the console boundaries
            - visualize the training state with a gauge bar.
            Maybe with a TensorFlow-like format:
            [====>......................]

    1)  ** Learning rate decay **
        eps_0: initial learning rate
        eps_t: learning rate at last epoch
        alpha: parameter that defines the (const) slope
               alpha = curr_epoch / total_epochs

    2)  ** Data shuffling **
        the original data is also shuffled because we passed it by reference
        this allows the user to see the current order of the data if needed

    3)  ** Momentum method **
        the drag parameter is currently set to a default value

    4)  ** Costs return **
        each cost function returns a Matrix<T> type with dim(1, n)
        where n might not be 1 if the prediction of the net is a vector
    **/

    //Matrix<T> _inputs = inputs;
    //Matrix<T> _labels = labels;

    /// if no specified final learning rate -> learning rate: const eps_0
    if(eps_t == -1)
        eps_t = m_lr;
    double curr_cost;
    double eps_0 = m_lr;
    /// variables instantiation for Matrix.vShuffle() algorithm
    bool shuffled{false};
    Matrix<T> dataset = _inputs.hStack(_labels);
    unsigned n_lines = _inputs.getRows();
    unsigned n_columns = _inputs.getCols();
    unsigned n_labels = _labels.getCols();
    /// instantiations for (mini) batch training
    Matrix<T> inputs_batch;
    Matrix<T> predictions_batch;
    Matrix<T> labels_batch;

    for(unsigned epoch{0}; epoch < epochs; epoch++){
        if(eps_t != -1){
            //std::cout << n << " " << d << " " << n/d << " " << m_lr << std::endl;
            double alpha = static_cast<double>(epoch) / static_cast<double>(epochs);
            this->m_lr = (1 - alpha) * eps_0 + alpha * eps_t;
        }
        for(unsigned batch{0}; batch < n_lines; batch+=m_batch_size){
            inputs_batch = _inputs.getSubmatrix(batch, 0, batch+m_batch_size, n_columns);
            predictions_batch = this->forwardPropagation(inputs_batch);
            //if(epoch % 1000 == 0)
            //    std::cout << "inputs" << _inputs << "labels" << _labels << "inputs" << *m_activationsMap["a"+std::to_string(0)] << "predictions" << predictions_batch << std::endl; /// DEBUGGING
            labels_batch = _labels.getSubmatrix(batch, 0, batch+m_batch_size, n_labels);
            this->momentum(predictions_batch, labels_batch, 0.5);
            //curr_cost = this->MSE(predictions_batch, labels_batch).hSum()(0, 0);
            //curr_cost = this->binaryCrossEntropy(predictions_batch, labels_batch).hSum()(0, 0);
            curr_cost = this->crossEntropy(predictions_batch, labels_batch).hSum()(0, 0); /// no need to feed labels in theory and practically
        }
        if(epoch % 5 == 0){
            for(double i{0}; i < curr_cost*50; i++)
                std::cout << "-";
            std::cout << "> (" << epoch << ")" << " (" << curr_cost << ") ";
            //std::cout << "shuffled: " << shuffled << " ";
            std::cout << "lr: " << m_lr << std::endl;

            shuffled = false;
        }
        /// dont shuffle at last epoch
        if(epoch < epochs-1){
            dataset = dataset.vShuffle();
            shuffled = true;
            _inputs = dataset.getSubmatrix(0, 0, n_lines, n_columns);
            _labels = dataset.getSubmatrix(0, n_columns, n_lines, n_columns+n_labels);
        }
    }


    /// print results after training
    /**
    std::cout << dataset << std::endl;
    for(unsigned it{0}; it < n_columns; it+=m_batch_size)
    {
        /// this is NOT real test data since the net has been trained with it
        Matrix<T> inputs_test = _inputs.getSubmatrix(it, 0, it+m_batch_size, n_columns);
        Matrix<T> predictions_test = this->forwardPropagation(inputs_test);
        std::cout << predictions_test << std::endl;
    }
    std::cout << std::endl;
    **/

    this->printErrors();
    this->printGradients();
    this->printWeights();
    this->printActivations();

}

template<typename T>
void FCANN<T>::printWeights() const
{
    std::cout << "------------------WEIGHTS------------------" << std::endl;
    std::map<std::string, Matrix<double>* >::const_iterator entry; //= myMap.begin(); //myMap.find("a3")
    unsigned i{1};
    for(auto entry = m_weightsMap.begin(); entry != m_weightsMap.end(); ++entry)
    {
        std::cout << "(" << i << ") dimensions w/ bias" << std::endl;
        std::cout << "(" << m_dimensionsVect[i-1].first << ", " << m_dimensionsVect[i-1].second << ")" << std::endl;
        if(entry != m_weightsMap.end())
        {
            std::string key = entry->first;
            Matrix<T> element = *entry->second;
            std::cout << key << std::endl;
            std::cout << element << std::endl;

            i++;
        }
    }
    std::cout << "COUNTER: " << VARIABLES<T>::COUNTER << std::endl;
    std::cout << "-------------------------------------" << std::endl;
}

template<typename T>
void FCANN<T>::printActivations() const
{
    std::cout << "------------------ACTIVATIONS------------------" << std::endl;
    std::map<std::string, Matrix<double>* >::const_iterator entry; //= myMap.begin(); //myMap.find("a3")
    unsigned i{1};
    for(auto entry = m_activationsMap.begin(); entry != m_activationsMap.end(); ++entry)
    {
        std::cout << "(" << i << ") dimensions w/ bias" << std::endl;
        std::cout << "(" << m_batch_size << ", " << m_dimensionsVect[i-1].second << ")" << std::endl;
        if(entry != m_activationsMap.end())
        {
            std::cout << entry->first << std::endl;   // key
            std::cout << *entry->second << std::endl; // element

            i++;
        }
    }
    std::cout << "COUNTER: " << VARIABLES<T>::COUNTER << std::endl;
    std::cout << "-------------------------------------" << std::endl;
}

template<typename T>
void FCANN<T>::printGradients() const
{
    std::cout << "------------------GRADIENTS------------------" << std::endl;
    std::map<std::string, Matrix<double>* >::const_iterator entry; //= myMap.begin(); //myMap.find("a3")
    unsigned i{1};
    for(auto entry = m_gradientsMap.begin(); entry != m_gradientsMap.end(); ++entry)
    {
        std::cout << "(" << i << ") dimensions w/ bias" << std::endl;
        std::cout << "(" << m_dimensionsVect[i-1].first << ", " << m_dimensionsVect[i-1].second << ")" << std::endl;
        if(entry != m_gradientsMap.end())
        {
            std::string key = entry->first;
            Matrix<T> element = *entry->second;
            std::cout << key << std::endl;
            std::cout << element << std::endl;

            i++;
        }
    }
    std::cout << "COUNTER: " << VARIABLES<T>::COUNTER << std::endl;
    std::cout << "-------------------------------------" << std::endl;
}

template<typename T>
void FCANN<T>::printErrors() const
{
    std::cout << "------------------ERRORS------------------" << std::endl;
    std::map<std::string, Matrix<double>* >::const_iterator entry; //= myMap.begin(); //myMap.find("a3")
    unsigned i{1};
    for(auto entry = m_errorsMap.begin(); entry != m_errorsMap.end(); ++entry)
    {
        std::cout << "(" << i << ") dimensions w/ bias" << std::endl;
        std::cout << "(" << m_batch_size << ", " << m_dimensionsVect[i-1].second << ")" << std::endl;
        if(entry != m_errorsMap.end())
        {
            std::string key = entry->first;
            Matrix<T> element = *entry->second;
            std::cout << key << std::endl;
            std::cout << element << std::endl;

            i++;
        }
    }
    std::cout << "COUNTER: " << VARIABLES<T>::COUNTER << std::endl;
    std::cout << "-------------------------------------" << std::endl;
}
