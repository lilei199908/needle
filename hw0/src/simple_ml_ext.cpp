#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_batches = (m + batch - 1) / batch; // ceiling division to cover all samples
    for (int b = 0; b < num_batches; ++b) {
        size_t start = b * batch;
        size_t end = std::min<size_t>(start + batch, m);
        size_t current_batch_size = end - start;
        // Allocate memory for logits and probabilities
        float* logits = new float[current_batch_size * k];
        float* probs = new float[current_batch_size * k];
        // Compute logits: logits = X_batch * theta
        for (size_t i = 0; i < current_batch_size; ++i) {
            for (size_t j = 0; j < k; ++j) {
                logits[i * k + j] = 0.0f;
                for (size_t l = 0; l < n; ++l) {
                    logits[i * k + j] += X[(start + i) * n + l] * theta[l * k + j];
                }
            }
        }
        // Apply softmax to logits to get probabilities
        for (size_t i = 0; i < current_batch_size; ++i) {
            float max_logit = logits[i * k];
            for (size_t j = 1; j < k; ++j) {
                if (logits[i * k + j] > max_logit) {
                    max_logit = logits[i * k + j];
                }
            }
            float sum_exp = 0.0f;
            for (size_t j = 0; j < k; ++j) {
                probs[i * k + j] = std::exp(logits[i * k + j] - max_logit);
                sum_exp += probs[i * k + j];
            }
            for (size_t j = 0; j < k; ++j) {
                probs[i * k + j] /= sum_exp;
            }
        }
        // Compute gradient and update theta
        for (size_t i = 0; i < current_batch_size; ++i) {
            for (size_t j = 0; j < k; ++j) {
                float indicator = (y[start + i] == j) ? 1.0f : 0.0f;
                float error = probs[i * k + j] - indicator;
                for (size_t l = 0; l < n; ++l) {
                    theta[l * k + j] -= lr * error * X[(start + i) * n + l] / current_batch_size;
                }
            }
        }
        // Free allocated memory
        delete[] logits;
        delete[] probs;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
