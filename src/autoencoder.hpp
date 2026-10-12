#pragma once
#include "linear.hpp"
#include "activations.hpp"
#include <random>

class Autoencoder{
    public:
        Linear enc1;
        Linear enc2;
        Linear dec1;
        Linear dec2;
        ReLU relu_enc;
        ReLU relu_dec;

        Autoencoder(std::mt19937& rng) : enc1(784, 400, rng), enc2(400, 2, rng), dec1(2, 400, rng), dec2(400, 784, rng) {}

        Matrix encode(const Matrix& X){
            Matrix a = enc1.forward(X);
            Matrix h = relu_enc.forward(a);
            return enc2.forward(h);
        }

        Matrix decode(const Matrix& Z){
            Matrix a = dec1.forward(Z);
            Matrix h = relu_dec.forward(a);
            return dec2.forward(h);
        }

        Matrix forward(const Matrix& X){
            Matrix Z = encode(X);
            return decode(Z);
        }

        void backward(const Matrix& d_scores){
            Matrix dh_dec = dec2.backward(d_scores);
            Matrix da_dec = relu_dec.backward(dh_dec);
            Matrix dZ = dec1.backward(da_dec);
            Matrix dh_enc = enc2.backward(dZ);
            Matrix da_enc = relu_enc.backward(dh_enc);
            enc1.backward(da_enc);
        }

        void adam_step(double lr, int t){
            enc1.adam_step(lr, t);
            enc2.adam_step(lr, t);
            dec1.adam_step(lr, t);
            dec2.adam_step(lr, t);
        }
};