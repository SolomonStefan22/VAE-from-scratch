#include "matrix.hpp"
#include "mnist.hpp"
#include "image.hpp"
#include "loss.hpp"
#include "activations.hpp"
#include "io.hpp"
#include "vae.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <cstdlib>
#include <cmath>



Matrix take_rows(const Matrix& X, const std::vector<size_t>& order, size_t start, size_t count){
    Matrix Bm(count, X.cols);
    for(size_t r = 0; r < count; ++r) {
        size_t src = order[start + r];
        for (size_t c = 0; c < X.cols; ++c)
            Bm.data[r * X.cols + c] = X.data[src * X.cols + c];
    }
    return Bm;
}

void shuffle(std::vector<size_t>& order, std::mt19937& rng){
    for(size_t i = order.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> pick(0, i);
        size_t j = pick(rng);
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

int main(int argc, char** argv) {
    size_t D = 2;
    if(argc > 1)
        D = (size_t)std::atoi(argv[1]);

    Matrix X_train = load_mnist_images("data/train-images-idx3-ubyte");
    Matrix X_test  = load_mnist_images("data/t10k-images-idx3-ubyte");
    std::vector<int> y_test = load_mnist_labels("data/t10k-labels-idx1-ubyte");

    std::mt19937 rng(0);
    VAE vae(D, rng);

    const size_t B = 100;
    const int epochs = 5;
    const double lr = 1e-3;

    std::vector<size_t> order(X_train.rows);
    for(size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::vector<size_t> test_order(X_test.rows);
    for(size_t i = 0; i < test_order.size(); ++i) test_order[i] = i;

    int t = 0;
    for(int epoch = 1; epoch <= epochs; ++epoch){
        shuffle(order, rng);
        size_t n_batches = X_train.rows / B;
        double sum_rec = 0.0, sum_kl = 0.0;

        for(size_t k = 0; k < n_batches; ++k){
            Matrix X = take_rows(X_train, order, k * B, B);
            Matrix scores = vae.forward(X, rng);
            double rec = bce_loss(scores, X);
            double kl  = vae.kl_loss();
            vae.backward(bce_grad(scores, X));
            t += 1;
            vae.adam_step(lr, t);
            sum_rec += rec;
            sum_kl  += kl;
        }

        double test_sum = 0.0;
        size_t test_batches = X_test.rows / B;
        for(size_t k = 0; k < test_batches; ++k){
            Matrix X = take_rows(X_test, test_order, k * B, B);
            Matrix scores = vae.forward(X, rng);
            test_sum += bce_loss(scores, X) + vae.kl_loss();
        }
        std::printf("Epoca %d: refacere %.2f, KL %.2f, total antrenare %.2f, total test %.2f\n\n", epoch, sum_rec / n_batches, sum_kl / n_batches, (sum_rec + sum_kl) / n_batches, test_sum / test_batches);
    }

    char name[64];
    Matrix X16 = take_rows(X_test, test_order, 0, 16);
    vae.encode(X16);
    Matrix R16 = sigmoid(vae.decode(vae.mu));
    Matrix both(32, 784);
    for(size_t k = 0; k < 16 * 784; ++k){
        both.data[k] = X16.data[k];
        both.data[16 * 784 + k] = R16.data[k];
    }

    std::snprintf(name, sizeof(name), "vae_reconstructii_d%zu.pgm", D);
    save_grid_pgm(name, both, 0, 32, 16);

    std::normal_distribution<double> normal(0.0, 1.0);
    Matrix Zs(64, D);
    for(size_t k = 0; k < Zs.data.size(); ++k) Zs.data[k] = normal(rng);
    
    std::snprintf(name, sizeof(name), "vae_generate_d%zu.pgm", D);
    save_grid_pgm(name, sigmoid(vae.decode(Zs)), 0, 64, 8);

    vae.encode(X_test);
    for(size_t k = 0; k < D; ++k){
        double s = 0.0;
        for(size_t r = 0; r < vae.mu.rows; ++r){
            double m = vae.mu.data[r * D + k];
            double vv = vae.v.data[r * D + k];
            s += 0.5 * (std::exp(vv) + m * m - 1.0 - vv);
        }
        std::printf("  dimensiunea %2zu: %.3f\n", k + 1, s / vae.mu.rows);
    }

    if(D == 2){
        save_scatter_ppm("vae_puncte.ppm", vae.mu, y_test, -4.5, 4.5, 600);
        const size_t n = 15;
        Matrix Zgrid(n * n, 2);
        for(size_t i = 0; i < n; ++i)
            for(size_t j = 0; j < n; ++j){
                Zgrid.data[(i * n + j) * 2]     = -3.0 + 6.0 * j / (n - 1);
                Zgrid.data[(i * n + j) * 2 + 1] =  3.0 - 6.0 * i / (n - 1);
            }
        save_grid_pgm("vae_harta.pgm", sigmoid(vae.decode(Zgrid)), 0, n * n, n);
    }

    std::snprintf(name, sizeof(name), "vae_d%zu.bin", D);
    save_vae(name, vae);

    return 0;
}