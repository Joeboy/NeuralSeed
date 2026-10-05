#pragma once

#include <stddef.h>

// PyTorch GRU weights, converted to the z/r/n layout used by RTNeural.
// Kept on the stack during loading; the processing path uses RTNeural itself.
struct NeuralSeedWeights {
    int inputs;
    int hidden;
    float wi[4][30];
    float wh[10][30];
    float bias[2][30];
    float dense[10];
    float dense_bias;
};

bool neuralseed_parse_json(const char* bytes, size_t size, NeuralSeedWeights* result);
