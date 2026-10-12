#include "matrix.hpp"
#include "mnist.hpp"
#include "image.hpp"
#include "loss.hpp"
#include "activations.hpp"
#include "autoencoder.hpp"
#include <cstdio>
#include <random>
#include <vector>

Matrix take_rows(const Matrix& X, const std::vector<size_t>& order, size_t start, size_t count){
    Matrix Bm(count, X.cols);
    for (size_t r = 0; r < count; ++r) {
        size_t src = order[start + r];
        for (size_t c = 0; c < X.cols; ++c)
            Bm.data[r * X.cols + c] = X.data[src * X.cols + c];
    }
    return Bm;
}

void shuffle(std::vector<size_t>& order, std::mt19937& rng){
    for (size_t i = order.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> pick(0, i);
        size_t j = pick(rng);
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
    }
}

int main(){
    Matrix X_train = load_mnist_images("data/train-images-idx3-ubyte");
    Matrix X_test  = load_mnist_images("data/t10k-images-idx3-ubyte");

    std::mt19937 rng(0);
    Autoencoder ae(rng);

    const size_t B = 100;
    const int epochs = 5;
    const double lr = 1e-3;

    std::vector<size_t> order(X_train.rows);
    for(size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::vector<size_t> test_order(X_test.rows);
    for(size_t i = 0; i < test_order.size(); ++i)
        test_order[i] = i;
    
    int t = 0;
    for(int epoch = 1; epoch <= epochs; ++epoch){
        shuffle(order, rng);
        size_t n_batches = X_train.rows / B;
        double sum = 0.0;

        for(size_t k = 0; k < n_batches; ++k){
            Matrix X = take_rows(X_train, order, k * B, B);
            Matrix scores = ae.forward(X);
            double cost = bce_loss(scores, X);
            ae.backward(bce_grad(scores, X));
            t += 1;
            ae.adam_step(lr, t);
            sum += cost;

            if(k == 0 && epoch == 1)
                std::printf("  primul grup: cost %.1f\n", cost);
            if((k + 1) % 100 == 0)
                std::printf("  epoca %d, grupul %zu/%zu, cost mediu %.2f\n", epoch, k + 1, n_batches, sum / (k + 1));
        }

        double test_sum = 0.0;
        size_t test_batches = X_test.rows / B;
        for(size_t k = 0; k < test_batches; ++k){
            Matrix X = take_rows(X_test, test_order, k * B, B);
            test_sum += bce_loss(ae.forward(X), X);
        }
        std::printf("Epoca %d: cost antrenare %.2f, cost test %.2f\n\n", epoch, sum / n_batches, test_sum / test_batches);
    }

    Matrix X16 = take_rows(X_test, test_order, 0, 16);
    Matrix R16 = sigmoid(ae.forward(X16));
    Matrix both(32, 784);
    for(size_t t2 = 0; t2 < 16 * 784; ++t2){
        both.data[t2] = X16.data[t2];
        both.data[16 * 784 + t2] = R16.data[t2];
    }
    save_grid_pgm("ae_reconstructii.pgm", both, 0, 32, 16);

    Matrix Z_all = ae.encode(X_test);
    double lo0 = Z_all.data[0], hi0 = Z_all.data[0];
    double lo1 = Z_all.data[1], hi1 = Z_all.data[1];
    for(size_t r = 0; r < Z_all.rows; ++r){
        double z0 = Z_all.data[r * 2], z1 = Z_all.data[r * 2 + 1];
        if (z0 < lo0) lo0 = z0;
        if (z0 > hi0) hi0 = z0;
        if (z1 < lo1) lo1 = z1;
        if (z1 > hi1) hi1 = z1;
    }
    std::printf("Codurile ocupa: z1 in [%.2f, %.2f], z2 in [%.2f, %.2f]\n", lo0, hi0, lo1, hi1);

    const size_t n = 15;
    Matrix Zgrid(n * n, 2);
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j){
            Zgrid.data[(i * n + j) * 2]     = lo0 + (hi0 - lo0) * j / (n - 1);
            Zgrid.data[(i * n + j) * 2 + 1] = hi1 - (hi1 - lo1) * i / (n - 1);
        }
    save_grid_pgm("ae_harta.pgm", sigmoid(ae.decode(Zgrid)), 0, n * n, n);
    std::printf("Am salvat ae_reconstructii.pgm si ae_harta.pgm\n");

    return 0;
}