#pragma once
#include <vector>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

inline void check(bool cond, const char* msg){
    if(!cond){
        std::printf("EROARE: %s\n", msg);
        std::exit(1);
    }
}

class Matrix{
    public:
        size_t rows;
        size_t cols;
        std::vector<double> data;

        Matrix(){
            rows = 0;
            cols = 0;
        }

        Matrix(size_t r, size_t c){
            rows = r;
            cols = c;
            data = std::vector<double>(r * c, 0);
        }

        Matrix(size_t r, size_t c, const std::vector<double>& values){
            check(values.size() == r * c, "Numarul de valori nu corespunde dimensiunilor");
            rows = r;
            cols = c;
            data = values;
        }

        double get(size_t i, size_t j) const{
            check(i < rows && j < cols, "indice in afara matricei");
            return data[i * cols + j];
        }

        void set(size_t i, size_t j, double value){
            check(i < rows && j < cols, "indice in afara matricei");
            data[i * cols + j] = value;
        }

        Matrix matmul(const Matrix& B) const{
            check(cols == B.rows, "dimensiuni incompatibile");
            Matrix C(rows, B.cols);
            for(size_t i = 0; i < rows; ++i){
                for(size_t k = 0; k < cols; ++k){
                    double a  = data[i * cols + k];
                    for(size_t j = 0; j < B.cols; ++j){
                        C.data[i * C.cols + j] += a * B.data[k * B.cols + j];
                    }
                }
            }

            return C;
        }

        Matrix transpose() const{
            Matrix T(cols, rows);
            for(size_t i = 0; i < rows; ++i)
                for(size_t j = 0; j < cols; ++j)
                    T.data[j * rows + i] = data[i * cols + j];
            return T;
        }

        Matrix add(const Matrix& other) const{
            check(rows == other.rows && cols == other.cols, "forme diferite");
            Matrix R(rows, cols);
            for(size_t t = 0; t < data.size(); ++t)
                R.data[t] = data[t] + other.data[t];
            return R;
        }

        Matrix sub(const Matrix& other) const{
            check(rows == other.rows && cols == other.cols, "forme diferite");
            Matrix R(rows, cols);
            for(size_t t = 0; t < data.size(); ++t)
                R.data[t] = data[t] - other.data[t];
            return R;
        }

        Matrix hadamard(const Matrix& other) const{
            check(rows == other.rows && cols == other.cols, "forme diferite");
            Matrix R(rows, cols);
            for(size_t t = 0; t < data.size(); ++t)
                R.data[t] = data[t] * other.data[t];
            return R;
        }

        Matrix scale(double s) const{
            Matrix R(rows, cols);
            for(size_t t = 0; t < data.size(); ++t)
                R.data[t] = data[t] * s;
            return R;
        }

        Matrix add_row(const Matrix& b) const{
            check(b.rows == 1 && b.cols == cols, "b trebuie sa fie vector de dim cols");
            Matrix R(rows, cols);
            for(size_t i = 0; i < rows; ++i)
                for(size_t j = 0; j < cols; ++j)
                    R.data[i * cols + j] = data[i * cols + j] + b.data[j];
            return R;
        }

        Matrix sum_rows() const{
            Matrix R(1, cols);
            for(size_t i = 0; i < rows; ++i)
                for(size_t j = 0; j < cols; ++j)
                    R.data[j] += data[i * cols + j];
            return R;
        }

        double sum() const{
            double s = 0.0;
            for(size_t t = 0; t < data.size(); ++t)
                s += data[t];
            return s;
        }
};