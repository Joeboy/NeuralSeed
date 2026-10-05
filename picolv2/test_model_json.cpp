#include "model_json.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static bool load(const char* path, NeuralSeedWeights* weights) {
    FILE* file = fopen(path, "rb");
    if (!file) return false;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    char* bytes = static_cast<char*>(malloc(size));
    bool ok = bytes && fread(bytes, 1, size, file) == static_cast<size_t>(size) &&
              neuralseed_parse_json(bytes, size, weights);
    free(bytes);
    fclose(file);
    return ok;
}

int main() {
    NeuralSeedWeights weights{};
    assert(load("../models/gru8_ts9_pytorch.json", &weights));
    assert(weights.inputs == 1 && weights.hidden == 8);
    assert(isfinite(weights.wi[0][0]));
    assert(load("../models/ts9_gru7_gain10_loss003_TEST.json", &weights));
    assert(weights.inputs == 1 && weights.hidden == 7);
    assert(!load("../models/lstm8_ts9_pytorch.json", &weights));
    const char* invalid = "{\"model_data\":{},\"state_dict\":{}}";
    assert(!neuralseed_parse_json(invalid, 33, &weights));
    puts("NeuralSeed JSON parser: OK");
}
