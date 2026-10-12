#pragma once
#include "linear.hpp"
#include "activations.hpp"
#include <random>
#include <cmath>

class VAE{
    public:
        size_t D;
        Linear enc1;
        Linear enc_mu;
        Linear enc_v;
        Linear dec1;
        Linear dec2;
        ReLU relu_enc;
        ReLU relu_dec;

        Matrix mu;
        Matrix v;
        Matrix z;

        VAE(size_t latent, std::mt19937& rng) : D(latent), enc1(784, 400, rng), enc_mu(400, latent, rng), enc_v(400, latent, rng), dec1(latent, 400, rng), dec2(400, 784, rng) {}

        void encode(const Matrix& X){
            Matrix a = enc1.forward(X);
            Matrix h = relu_enc.forward(a);
            mu = enc_mu.forward(h);
            v = enc_v.forward(h);
        }

        Matrix decode(const Matrix& Z){
            Matrix a = dec1.forward(Z);
            Matrix h = relu_dec.forward(a);
            return dec2.forward(h);
        }

        Matrix forward(const Matrix& X, std::mt19937& rng){
            encode(X);
            std::normal_distribution<double> normal(0.0, 1.0);
            z = Matrix(mu.rows, mu.cols);
            for(size_t t = 0; t < z.data.size(); ++t){
                double eps = normal(rng);
                z.data[t] = mu.data[t] + std::exp(0.5 * v.data[t]) * eps;
            }
            return decode(z);
        }

        double kl_loss() const{
            double s = 0.0;
            for(size_t t = 0; t < mu.data.size(); ++t)
                s += 0.5 * (std::exp(v.data[t]) + mu.data[t] * mu.data[t] - 1.0 - v.data[t]);
            return s / mu.rows;
        }

        void backward(const Matrix& d_scores){
            Matrix dh_dec = dec2.backward(d_scores);
            Matrix da_dec = relu_dec.backward(dh_dec);
            Matrix dz = dec1.backward(da_dec);

            double B = (double)mu.rows;
            Matrix dmu(mu.rows, mu.cols);
            Matrix dv(v.rows, v.cols);
            for(size_t t = 0; t < mu.data.size(); ++t){
                dmu.data[t] = dz.data[t] + mu.data[t] / B;
                dv.data[t] = dz.data[t] * 0.5 * (z.data[t] - mu.data[t]) + 0.5 * (std::exp(v.data[t]) - 1.0) / B;
            }

            Matrix dh_from_mu = enc_mu.backward(dmu);
            Matrix dh_from_v = enc_v.backward(dv);
            Matrix dh_enc = dh_from_mu.add(dh_from_v);

            Matrix da_enc = relu_enc.backward(dh_enc);
            enc1.backward(da_enc);
        }

        void adam_step(double lr, int t){
            enc1.adam_step(lr, t);
            enc_mu.adam_step(lr, t);
            enc_v.adam_step(lr, t);
            dec1.adam_step(lr, t);
            dec2.adam_step(lr, t);
        }
};