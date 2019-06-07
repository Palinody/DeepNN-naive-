#include <iostream>
#include "Matrix.h"

template class Matrix<int>;
template class Matrix<unsigned>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<char>;

template <typename T>
unsigned VARIABLES<T>::COUNTER = 0;

template<typename T>
Matrix<T>::Matrix(){
    VARIABLES<T>::COUNTER++;
}

template<typename T>
Matrix<T>::Matrix(unsigned rows, unsigned columns, const T& initValue) : m_rows(rows), m_columns(columns){
    m_matrix.resize(m_rows);
    for(unsigned i(0); i < m_matrix.size(); i++){
        m_matrix[i].resize(m_columns, initValue);
    }
    VARIABLES<T>::COUNTER++;
}

template<typename T>
Matrix<T>::Matrix(unsigned rows, unsigned columns, std::string distribution, \
                  double mean, double sigma, \
                  double inf, double sup) : m_rows{rows}, m_columns{columns}{
    //get gaussian distribution matrix
    // random device class instance, source of 'true' randomness for initializing random seed
    ///std::random_device rd{};
    // Mersenne twister PRNG, initialized with seed from previous random device instance
    ///std::mt19937 generator{rd()};

    // converting time to long int
    long int current_time{static_cast<long int>(time(NULL))};
    time_t _seed{static_cast<time_t>(VARIABLES<T>::COUNTER + current_time)};
    std::mt19937 generator{_seed};

    if(distribution == "normal" || distribution == "gaussian"){
        std::normal_distribution<double> distr1(mean, sigma);
        m_matrix.resize(m_rows);
        for(unsigned i{0}; i < m_matrix.size(); i++){
            m_matrix[i].resize(m_columns, distr1(generator));
            for(unsigned j{0}; j < m_columns; j++)
                m_matrix[i][j] = distr1(generator);
        }
    }
    else
     if(distribution == "uniform"){
        std::uniform_real_distribution<double> distr1(inf, sup);
        m_matrix.resize(m_rows);
        for(unsigned i{0}; i < m_matrix.size(); i++){
            m_matrix[i].resize(m_columns, 0);
            for(unsigned j{0}; j < m_columns; j++)
                m_matrix[i][j] = distr1(generator);
        }
     }
     else
     if(distribution == "Xavier"){
        mean = 0; sigma = sqrt(2 / (m_rows + m_columns - 1 + 1e-8)); // subtract 1 because of bias term
        std::normal_distribution<double> distr1(mean, sigma);
        m_matrix.resize(m_rows);
        for(unsigned i{0}; i < m_matrix.size(); i++){
            m_matrix[i].resize(m_columns, distr1(generator));
            for(unsigned j{0}; j < m_columns; j++)
                m_matrix[i][j] = distr1(generator);
        }
     }
     VARIABLES<T>::COUNTER++;
}

//Copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other){
    m_matrix = other.m_matrix;
    m_rows = other.getRows();
    m_columns = other.getCols();

    VARIABLES<T>::COUNTER++;
}

//Virtual destructor
template<typename T>
Matrix<T>::~Matrix(){
    --VARIABLES<T>::COUNTER;
}

template<typename T>
void Matrix<T>::eye(){
    for(unsigned i{0}; i < m_rows; i++)
        for(unsigned j{0}; j < m_columns; j++)
            if(i == j)
                this->m_matrix[i][j] = 1;
}

//Return diagonal elements of a Matrix
template<typename T>
std::vector<T> Matrix<T>::getDiagonalVect(){
    if(m_rows <= m_columns){
        std::vector<T> resultVector(m_rows, 0.0);
        for(unsigned i(0); i < m_rows; i++){
            resultVector[i] = this->m_matrix[i][i];
        }
        return resultVector;
    }
    std::vector<T> resultVector(m_columns, 0.0);
    for(unsigned i(0); i < m_columns; i++){
        resultVector[i] = this->m_matrix[i][i];
    }
    return resultVector;
}

//Return diagonal elements of a Matrix
template<typename T>
Matrix<T> Matrix<T>::getDiagonal(){
    if(m_rows <= m_columns){
        Matrix<T> resultVector(1, m_rows, 0.0);
        for(unsigned i(0); i < m_rows; i++){
            resultVector(0, i) = this->m_matrix[i][i];
        }
        return resultVector;
    }
    Matrix<T> resultVector(1, m_columns, 0.0);
    for(unsigned i(0); i < m_columns; i++){
        resultVector(0, i) = this->m_matrix[i][i];
    }
    return resultVector;
}

template<typename T>
Matrix<T> Matrix<T>::getSubmatrix(unsigned from_i, unsigned from_j, unsigned to_i, unsigned to_j){
    unsigned sub_rows = to_i - from_i;
    unsigned sub_columns = to_j - from_j;
    Matrix<T> submatrix(sub_rows, sub_columns, 0);

    unsigned sub_i{0};
    for(unsigned i{from_i}; i < to_i; i++){
        unsigned sub_j{0};
        for(unsigned j{from_j}; j < to_j; j++){
            submatrix(sub_i, sub_j) = this->m_matrix[i][j];
            sub_j++;
        }
        sub_i++;
    }
    return submatrix;
}

template<typename T>
void Matrix<T>::insertFrame(Matrix<T>& other,\
                        unsigned from_i, unsigned from_j){
    try
    {
        if((from_i + other.getRows() > this->getRows()) || (from_j + other.getCols() > this->getCols())){
            throw std::string("Frame is out of bounds.");
        }
        else{
            for(unsigned i{0}; i < other.getRows(); i++)
                for(unsigned j{0}; j < other.getCols(); j++)
                    this->m_matrix[from_i+i][from_j+j] = other(i, j);
        }
    }
    catch(const Matrix<T>& matrix) {std::cerr << matrix << std::endl;}
}

//Transpose of this matrix
template<typename T>
Matrix<T> Matrix<T>::transpose(){
    Matrix<T> cacheMatrix(m_columns, m_rows, 0.0);
    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            cacheMatrix(j, i) = this->m_matrix[i][j];
    return cacheMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::self_transpose(){
    Matrix<T> cacheMatrix(m_columns, m_rows, 0.0);
    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            cacheMatrix(j, i) = this->m_matrix[i][j];
    *this = cacheMatrix;
    return *this;
}

//Horizontal concatenation of two matrices
template<typename T>
Matrix<T> Matrix<T>::hStack(const Matrix<T>& other){
    /*
    Future improvement:
        directly set stackedMatrix[0:m_rows][0:m_columns] = m_matrix
    and
        directly set stackedMatrix[0:m_rows][m_columns+1:m_columns+other.getCols()]
    */
    unsigned new_rows{m_rows};
    unsigned new_columns{m_columns + other.getCols()};
    Matrix<T> stackedMatrix(new_rows, new_columns, 0);
    for(unsigned i{0}; i < new_rows; i++){
        for(unsigned j{0}; j < m_columns; j++)
            stackedMatrix(i, j) = this->m_matrix[i][j];

        for(unsigned j{0}; j < other.getCols(); j++)
            stackedMatrix(i, m_columns + j) = other(i, j);
    }
    return stackedMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::vStack(const Matrix<T>& other){
    /*
    Future improvement:
        directly set stackedMatrix[0:m_rows][0:m_columns] = m_matrix
    and
        directly set stackedMatrix[0:m_rows][m_columns+1:m_columns+other.getCols()]
    */
    unsigned new_rows{m_rows + other.getRows()};
    unsigned new_columns{m_columns};
    Matrix<T> stackedMatrix(new_rows, new_columns, 0);

    for(unsigned i{0}; i < m_rows; i++)
        for(unsigned j{0}; j < m_columns; j++)
            stackedMatrix(i, j) = this->m_matrix[i][j];

    for(unsigned i{0}; i < other.getRows(); i++)
        for(unsigned j{0}; j < m_columns; j++)
            stackedMatrix(m_rows + i, j) = other(i, j);

    return stackedMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::hSum(){
    Matrix<T> summedMatrix(m_rows, 1, 0);
    for(unsigned i{0}; i < m_rows; i++){
        for(unsigned j{0}; j < m_columns; j++){
            summedMatrix(i, 0) += this->m_matrix[i][j];
        }
    }
    return summedMatrix;
}

template<typename T>
Matrix<T> Matrix<T>::vSum(){
    Matrix<T> summedMatrix(1, m_columns, 0);
    for(unsigned i{0}; i < m_rows; i++)
        for(unsigned j{0}; j < m_columns; j++)
            summedMatrix(0, j) += this->m_matrix[i][j];
    return summedMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::vShuffle(){
    /** Shuffles data vectors line wise **/
    std::vector<unsigned> indices;
    indices.reserve(m_rows);

    for(unsigned i{0}; i < m_rows; ++i)
        indices.push_back(i);
    std::random_shuffle(indices.begin(), indices.end());

    std::vector<std::vector<T>> tempMatrix;
    tempMatrix.resize(m_rows);
    for(unsigned i{0}; i < m_rows; i++)
        tempMatrix[i].resize(m_columns, 0);

    unsigned tempRow{0};
    for(std::vector<unsigned>::iterator it = indices.begin(); it != indices.end(); ++it){
        for(unsigned j{0}; j < m_columns; j++){
            tempMatrix[tempRow][j] = m_matrix[*it][j];
        }
        tempRow++;
    }
    /// put back tempMatrix content into m_matrix
    for(unsigned i{0}; i < m_rows; i++)
        m_matrix[i].swap(tempMatrix[i]);

    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::activationFunction(const std::string& activation_func){
    if(activation_func == "softmax"){
        Matrix<T> new_layer = *this;
        new_layer = (new_layer.activationFunction("exp")).hSum();
        Activation<T> activation(activation_func, new_layer, this->getRows(), this->getCols());
        std::for_each(Vector2D_iterator<T>::begin(m_matrix), Vector2D_iterator<T>::end(m_matrix), activation);
    }
    else{
        Activation<T> activation(activation_func);
        std::for_each(Vector2D_iterator<T>::begin(m_matrix), Vector2D_iterator<T>::end(m_matrix), activation);
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::deriveActivation(const std::string& activation_func){
    ActivationDerivative<T> activation_der(activation_func);
    std::for_each(Vector2D_iterator<T>::begin(m_matrix), Vector2D_iterator<T>::end(m_matrix), activation_der);
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::getActivationDerivative(const std::string& activation_func){
    Matrix<T> tempMatrix = (*this); /// keeps old instance intact
    this->deriveActivation(activation_func);
    Matrix<T> retMatrix = (*this);  /// returned matrix = this->derived
    (*this) = tempMatrix;           /// *this turns back to normal
    return retMatrix;
}

template<typename T>
Matrix<T>& Matrix<T>::elemProduct(const Matrix<T>& other){
    /// !!! try, catch, throw... or assert(..., ...) to manage matrices dimensions !!!
    for(unsigned i{0}; i < m_rows; i++)
        for(unsigned j{0}; j < m_columns; j++)
            m_matrix[i][j] *= other(i, j);
    return *this;
}

// horizontal product broadcasting Matrix - Vector
template<typename T>
Matrix<T>& Matrix<T>::hProdBroadcast(const Matrix<T>& other){
    /// assert(this->m_rows == other.get_rows());
    for(unsigned i{0}; i < this->m_rows; i++)
        for(unsigned j{0}; j < this->m_columns; j++)
            this->m_matrix[i][j] *= other(i, 0);
    return *this;
}

// vertical product broadcasting Matrix - Vector
template<typename T>
Matrix<T>& Matrix<T>::vProdBroadcast(const Matrix<T>& other){
    /// assert(this->m_columns == other.get_cols());
    for(unsigned j{0}; j < this->m_columns; j++)
        for(unsigned i{0}; i < this->m_rows; i++)
            this->m_matrix[i][j] *= other(0, j);
    return *this;
}

//Assignment operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other){
    if(&other == this)
        return *this;
    unsigned new_rows = other.getRows();
    unsigned new_cols = other.getCols();

    m_matrix.resize(new_rows);
    for(unsigned i(0); i < m_matrix.size(); i++){
        m_matrix[i].resize(new_cols);
    }
    for(unsigned i(0); i < new_rows; i++){
        for(unsigned j(0); j < new_cols; j++){
            m_matrix[i][j] = other(i, j);
        }
    }
    m_rows = new_rows;
    m_columns = new_cols;

    return *this;
}

//Addition of two matrices
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other){
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++){
        for(unsigned j(0); j < m_columns; j++){
            resultMatrix(i, j) = this->m_matrix[i][j] + other(i, j);
        }
    }
    return resultMatrix;
}

//Cumulative addition of this matrix and other matrix
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other){
    unsigned rows = other.getRows();
    unsigned cols = other.getCols();

    for(unsigned i(0); i < rows; i++)
        for(unsigned j(0); j < cols; j++)
            this->m_matrix[i][j] += other(i, j);
    return *this;
}

//Substraction of two matrices
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other)
{
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            resultMatrix(i, j) = this->m_matrix[i][j] - other(i, j);
    return resultMatrix;
}

//Cumulative subtraction of this matrix and other Matrix
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other){
    unsigned rows = other.getRows();
    unsigned cols = other.getCols();

    for(unsigned i(0); i < rows; i++)
        for(unsigned j(0); j < cols; j++)
            this->m_matrix[i][j] -= other(i, j);
    return *this;
}

//Left multiplication of this matrix and other matrix
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other){
    unsigned new_rows = m_rows;
    unsigned new_cols = other.getCols();

    Matrix<T> resultMatrix(new_rows, new_cols, 0.0);

    for(unsigned i(0); i < new_rows; i++)
        for(unsigned j(0); j < new_cols; j++)
            for(unsigned k(0); k < m_columns; k++)
                resultMatrix(i, j) += this->m_matrix[i][k] * other(k, j);
    return resultMatrix;
}

//Cumulative left multiplication of this matrix and other matrix
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& other){
    Matrix<T> resultMatrix = (*this) * other;
    (*this) = resultMatrix;
    return *this;
}

//elementwise division of two matrices
template<typename T>
Matrix<T> Matrix<T>::operator/(const Matrix<T>& other){
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            resultMatrix(i, j) = this->m_matrix[i][j] / other(i, j);
    return resultMatrix;
}

//Cumulative left division of this matrix and other matrix
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& other){
    Matrix<T> resultMatrix = (*this) / other;
    (*this) = resultMatrix;
    return *this;
}

//Matrix - scalar addition
template<typename T>
Matrix<T> Matrix<T>::operator+(const T& scalar){
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            resultMatrix(i, j) += this->m_matrix[i][j] + scalar;
    return resultMatrix;
}

//Matrix - scalar subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const T& scalar){
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            resultMatrix(i, j) += this->m_matrix[i][j] - scalar;
    return resultMatrix;
}

//Matrix - scalar multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar){
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            resultMatrix(i, j) += this->m_matrix[i][j] * scalar;
    return resultMatrix;
}

//Matrix - scalar division
template<typename T>
Matrix<T> Matrix<T>::operator/(const T& scalar){
    Matrix<T> resultMatrix(m_rows, m_columns, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned j(0); j < m_columns; j++)
            resultMatrix(i, j) += this->m_matrix[i][j] / scalar;
    return resultMatrix;
}

// scalar - Matrix addition affectation
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const T& scalar){
    for(unsigned i{0}; i < this->getRows(); i++)
        for(unsigned j{0}; j < this->getCols(); j++)
            this->m_matrix[i][j] = this->m_matrix[i][j] + scalar;
    return *this;
}

// scalar - Matrix multiplication affectation
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T& scalar){
    for(unsigned i{0}; i < this->getRows(); i++)
        for(unsigned j{0}; j < this->getCols(); j++)
            this->m_matrix[i][j] = this->m_matrix[i][j] * scalar;
    return *this;
}

// scalar - Matrix subtraction affectation
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const T& scalar){
    for(unsigned i{0}; i < this->getRows(); i++)
        for(unsigned j{0}; j < this->getCols(); j++)
            this->m_matrix[i][j] = scalar - this->m_matrix[i][j];
    return *this;
}

// scalar - Matrix division affectation
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T& scalar){
    for(unsigned i{0}; i < this->getRows(); i++)
        for(unsigned j{0}; j < this->getCols(); j++)
            this->m_matrix[i][j] = scalar / this->m_matrix[i][j];
    return *this;
}

//Matrix - vector multiplication
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& other){
    std::vector<T> resultVector(m_rows, 0.0);

    for(unsigned i(0); i < m_rows; i++)
        for(unsigned k(0); k < m_columns; k++)
            resultVector[i] += this->m_matrix[i][k] * other[k];
    return resultVector;
}

//Access the individual elements of this matrix
template<typename T>
T& Matrix<T>::operator()(const unsigned& row, const unsigned& column){
    return this->m_matrix[row][column];
}

//Access the individual elements (const)
template<typename T>
const T& Matrix<T>::operator()(const unsigned& row, const unsigned& column) const{
    return this->m_matrix[row][column];
}

//Get number of rows in this matrix
template<typename T>
unsigned Matrix<T>::getRows() const{
    return this->m_rows;
}

//Get number of columns in this matrix
template<typename T>
unsigned Matrix<T>::getCols() const{
    return this->m_columns;
}

template<typename T>
void Matrix<T>::print() const{
    int longer_string{0};
    std::ostringstream strs;

    for(unsigned i{0}; i < m_rows; i++){
        for(unsigned j{0}; j < m_columns; j++){
            strs << m_matrix[i][j];
            std::string str = strs.str();
            int current_size = str.size();
            if(current_size > longer_string){
                longer_string = current_size;
            }
        }
    }
    std::cout << "Matrix dimensions: (" << m_rows << ", " << m_columns << ")" << std::endl;
    std::cout << "[";
    for(unsigned i{0}; i < m_rows; i++){
        if(i == 0)
            std::cout << "[";
        else
            std::cout << " [";
        for(unsigned j{0}; j < m_columns; j++){
            strs << m_matrix[i][j];
            std::string str = strs.str();
            int current_size = str.size();
            std::cout << m_matrix[i][j];
            int steps{longer_string - current_size};
            for(int k{0}; k < steps; k++){
                std::cout << " ";
            }
            //handles end of row
            if(j < m_columns-1)
                std::cout << ", ";
            else
                std::cout << " ";
        }
        //handle very bottom right square brackets of the 2D-array
        if(i < m_rows-1)
            std::cout << "]," << std::endl;
        else
            std::cout << "]]" << std::endl;
    }
}

template<typename T>
void Matrix<T>::print(std::ostream& STREAM) const{
    int longer_string{0};
    std::ostringstream strs;

    for(unsigned i{0}; i < m_rows; i++){
        for(unsigned j{0}; j < m_columns; j++){
            strs << m_matrix[i][j];
            std::string str = strs.str();
            int current_size = str.size();
            if(current_size > longer_string){
                longer_string = current_size;
            }
        }
    }
    STREAM << "Matrix dimensions: (" << m_rows << ", " << m_columns << ")" << std::endl;
    STREAM << "[";
    for(unsigned i{0}; i < m_rows; i++){
        if(i == 0)
            STREAM << "[";
        else
            STREAM << " [";
        for(unsigned j{0}; j < m_columns; j++){
            strs << m_matrix[i][j];
            std::string str = strs.str();
            int current_size = str.size();
            STREAM << m_matrix[i][j];
            int steps{longer_string - current_size};
            for(int k{0}; k < steps; k++){
                STREAM << " ";
            }
            //handles end of row
            if(j < m_columns-1)
                STREAM << ", ";
            else
                STREAM << " ";
        }
        //handle very bottom right square brackets of the 2D-array
        if(i < m_rows-1)
            STREAM << "]," << std::endl;
        else
            STREAM << "]]" << std::endl;
    }
}
