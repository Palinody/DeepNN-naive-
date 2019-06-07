#include<iostream>
#include<sstream> //used to convert double to string
#include<string>
#include<map>
#include <fstream>
#include <math.h> /* nearbyint */

#include "Matrix.h"
#include "FCANN.h"
/*
COMPILE WITH VISUAL STUDIO CODE
https://stackoverflow.com/questions/30269449/how-do-i-set-up-visual-studio-code-to-compile-c-code

**HEADERS
//https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Header-File
**SOURCE CODE
//https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Source-File


**BOOST
//https://www.boost.org/
**BOOST/FULL DOCUMENTATION
//https://www.boost.org/doc/libs/1_68_0/
**BOOST/MATH
//https://www.boost.org/doc/libs/1_68_0/?view=category_math

**GAUSSIAN DISTRIBUTION WITH BOOST
//https://stackoverflow.com/questions/2078474/how-to-use-boost-normal-distribution-classes
*/

void getFileDimensions(std::string path, unsigned& n_lines, unsigned& n_columns){
    /*get number of lines in the file*/
    std::ifstream myFlux(path.c_str());
    std::string line;
    // we store number of columns and check
    // if number of columns is consistend
    unsigned total_elements{0};

    if(myFlux){
        while(getline(myFlux, line)){
            n_lines++;
            // count number of columns
            std::istringstream iss(line);
            n_columns = 0;
            do{
                std::string sub;
                iss >> sub;
                if(sub.length())
                    n_columns++;
            } while(iss);
            total_elements += n_columns;
        }
        n_columns = total_elements / n_lines; //nearbyint
        std::cout << "Done" << std::endl;
    }
    else
        std::cout << "ERROR" << std::endl;
    //return n_columns / (total_elements / n_lines);
}

template<typename T>
void fillData(Matrix<T>& matrix, std::string fromPath){
    /**
    get data from file to matrix
    I need to find a way to handle missing values
    **/
    std::ifstream myFlux(fromPath.c_str());

    if(myFlux){
        for(unsigned i{0}; i < matrix.getRows(); i++)
            for(unsigned j{0}; j < matrix.getCols(); j++)
                myFlux >> matrix(i, j);
    }
    else
        std::cout << "ERROR" << std::endl;
}

template<typename T>
void addData(const std::string& path, Matrix<T> data){
    std::ofstream myFlux(path.c_str(), std::ios::app); //app -> append at the end of file instead of erase and create
    if(myFlux)
    {
        //myFlux << line << endl;
        for(unsigned i{0}; i < data.getRows(); i++)
        {
            for(unsigned j{0}; j < data.getCols(); j++)
            {
                myFlux << data(i, j);
                myFlux << ", ";
            }
            myFlux << "\n";
            if(i == data.getRows()-1)
                std::cout << printf("%8d\r", i);
        }
        std::cout << std::endl;
    }
    else
        std::cout << "some ERROR occurred with the file" << std::endl;
}

void quick_training(){
    /* simple logic gates based examples data */
    // data file path
    ///std::string from_path1{"data/xor_data.txt"};
    ///std::string from_path1{"data/my_data.csv"};
    ///std::string from_path1{"data/xor_data_class.txt"};
    ///std::string from_path1{"data/logic_gates_classification.txt"};
    std::string from_path1{"data/logic_gates_classification_vectorized_simplified_again.txt"};
    unsigned n_lines{0}, n_columns{0};

    getFileDimensions(from_path1, n_lines, n_columns);

    Matrix<double> dataset(n_lines, n_columns, 0);
    // referenced matrix now holds inputs + labels

    fillData(dataset, from_path1);
    std::cout << "DATASET:\n" << dataset << std::endl;
    dataset.vShuffle();
    std::cout << "SHUFFLED DATASET:\n" << dataset << std::endl;

    // if we know the number of labels per element
    unsigned n_labels{6};
    Matrix<double> inputs = dataset.getSubmatrix(0, 0, n_lines, n_columns-n_labels);
    Matrix<double> labels = dataset.getSubmatrix(0, n_columns-n_labels, n_lines, n_columns);
    std::cout << "LABELS: \n" << labels << std::endl;
    /// FCANN TESTS
    std::vector<unsigned> dimVect = {5, 4, 3, 4, 5, 6}; //bottleneck auto-encoder test
    //std::vector<unsigned> dimVect = {6, 8, 6, 4};
    //std::vector<unsigned> dimVect = {80, 80, 80, 80, 80, 80, 80, 80, 80};
    //std::vector<unsigned> dimVect = {20};
    unsigned mini_batch = n_lines;
    double lr_0{0.1};  /// initial learning rate
    double lr_t{0.001};/// final learning rate
    FCANN<double> ann(n_columns-n_labels, n_labels, mini_batch, lr_0, dimVect);
    ann.train(inputs, labels, 10000, lr_t);

    std::cout << "we are here" << std::endl;
    dataset.vShuffle();
    inputs = dataset.getSubmatrix(0, 0, \
                                  n_lines, n_columns-n_labels);
    labels = dataset.getSubmatrix(0, n_columns-n_labels, \
                                  n_lines, n_columns);
    std::cout << dataset << std::endl;
    std::cout << inputs << std::endl;
    std::cout << labels << std::endl;
    for(unsigned it{0}; it < n_lines; it+=mini_batch){
        Matrix<double> input_test = inputs.getSubmatrix(it, 0, it+mini_batch, n_columns-n_labels);
        Matrix<double> prediction_test = ann.forwardPropagation(input_test);
        std::cout << input_test << std::endl;
        std::cout << labels << std::endl;
        std::cout << prediction_test << std::endl;
    }
    std::cout << std::endl;
    ann.printActivations();
}

void mnist_training(){
    /*
    Getting data from file to a data matrix
    and splitting it into two separate matrix: inputs and labels
    */
    // data file path
    //std::string abs_path{"C:/Users/lukar/Desktop/c_cpp_projects/CPP/projects/matrix/Python/MNIST/cpp_data/normalized_without_zeros/"};
    std::string abs_path{"Python/MNIST/cpp_data/normalized_without_zeros/"};

    unsigned n_lines_train_data{0}, n_columns_train_data{0};
    unsigned n_lines_test_data{0}, n_columns_test_data{0};
    unsigned n_lines_train_labels{0}, n_columns_train_labels{0};
    unsigned n_lines_test_labels{0}, n_columns_test_labels{0};

    std::string path1 = abs_path;
    std::string path2 = abs_path;
    std::string path3 = abs_path;
    std::string path4 = abs_path;

    path1.append("train_data_imgs.txt");
    path2.append("test_data_imgs.txt");
    path3.append("train_labels_oh.txt");
    path4.append("test_labels_oh.txt");

    getFileDimensions(path1, n_lines_train_data, n_columns_train_data);
    getFileDimensions(path2, n_lines_test_data, n_columns_test_data);
    getFileDimensions(path3, n_lines_train_labels, n_columns_train_labels);
    getFileDimensions(path4, n_lines_test_labels, n_columns_test_labels);

    ///**
    Matrix<double> training_data(n_lines_train_data, n_columns_train_data, 0);
    Matrix<double> test_data(n_lines_test_data, n_columns_test_data, 0);
    Matrix<double> training_labels(n_lines_train_labels, n_columns_train_labels, 0);
    Matrix<double> test_labels(n_lines_test_labels, n_columns_test_labels, 0);
    // referenced matrix now holds data
    fillData(training_data, path1);
    fillData(test_data, path2);
    fillData(training_labels, path3);
    fillData(test_labels, path4);

    /// FCANN TESTS
    //std::vector<unsigned> dimVect = {5, 4, 3, 4, 5, 6}; //bottleneck auto-encoder test
    std::vector<unsigned> dimVect = {20};
    unsigned mini_batch = n_lines_train_data / 400; /// 10000/400 = 25; 60000/400 = 150
    double lr_0{0.1};  /// initial learning rate
    double lr_t{0.001};/// final learning rate
    FCANN<double> ann(n_columns_train_data, n_columns_train_labels, mini_batch, lr_0, dimVect);
    ann.train(training_data, training_labels, 10, lr_t);


    std::cout << "WE ARE HERE" << std::endl;
    for(unsigned it{0}; it < n_lines_test_data; it+=mini_batch){
        Matrix<double> input_test = test_data.getSubmatrix(it, 0, it+mini_batch, n_columns_test_data);
        Matrix<double> prediction_test = ann.forwardPropagation(input_test);
        std::cout << input_test << std::endl;
        std::cout << test_labels.getSubmatrix(it, 0, it+mini_batch, n_columns_test_labels) << std::endl;
        std::cout << prediction_test << std::endl;
    }
    std::cout << std::endl;
    ann.printActivations();
    //**/
    ///addData("data/savedData.csv", dataset);
}

int main()
{

    quick_training();
    //mnist_training();
    return 0;
}
