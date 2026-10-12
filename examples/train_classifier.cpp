#include "matrix.hpp"
#include "mnist.hpp"
#include "loss.hpp"
#include "batch.hpp"
#include "classifier.hpp"
#include "io.hpp"
#include <cstdio>
#include <random>
#include <vector>

double accuracy(Classifier& clf, const Matrix& X, const std::vector<int>& y){
    std::vector<int> pred = argmax_rows(clf.forward(X));
    size_t correct = 0;
    for (size_t i = 0; i < y.size(); ++i)
        if (pred[i] == y[i]) correct++;
    return 100.0 * correct / y.size();
}

int main() {
    Matrix X_train = load_mnist_images("data/train-images-idx3-ubyte");
    std::vector<int> y_train = load_mnist_labels("data/train-labels-idx1-ubyte");
    Matrix X_test = load_mnist_images("data/t10k-images-idx3-ubyte");
    std::vector<int> y_test = load_mnist_labels("data/t10k-labels-idx1-ubyte");

    std::mt19937 rng(0);
    Classifier clf(rng);

    const size_t B = 100;
    const int epochs = 3;
    const double lr = 1e-3;

    std::vector<size_t> order(X_train.rows);
    for(size_t i = 0; i < order.size(); ++i) order[i] = i;

    int t = 0;
    for(int epoch = 1; epoch <= epochs; ++epoch){
        shuffle(order, rng);
        size_t n_batches = X_train.rows / B;
        double sum = 0.0;

        for(size_t k = 0; k < n_batches; ++k){
            Matrix X = take_rows(X_train, order, k * B, B);
            std::vector<int> y = take_labels(y_train, order, k * B, B);

            Matrix scores = clf.forward(X);
            double cost = softmax_ce_loss(scores, y);
            clf.backward(softmax_ce_grad(scores, y));
            t += 1;
            clf.adam_step(lr, t);
            sum += cost;
        }

        std::printf("Epoca %d: cost %.4f, corect pe test %.2f%%\n", epoch, sum / n_batches, accuracy(clf, X_test, y_test));
    }

    save_classifier("classifier.bin", clf);

    return 0;
}