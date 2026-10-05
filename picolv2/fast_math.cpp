#include "fast_math.h"
#include "fast_tanh_table.h"

#include <math.h>

float neuralseed_fast_tanh(float x) noexcept {
    const bool negative = x < 0.0f;
    const float scaled = fabsf(x) * 64.0f;
    if (scaled >= 512.0f) return negative ? -1.0f : 1.0f;
    const unsigned index = static_cast<unsigned>(scaled);
    const float fraction = scaled - static_cast<float>(index);
    const float value = kTanhTable[index] + fraction *
        (kTanhTable[index + 1] - kTanhTable[index]);
    return negative ? -value : value;
}

// The linker wraps RTNeural's std::tanh(float) call on Pico. The normal PC
// build keeps libm's tanhf for an exact reference.
extern "C" float __wrap_tanhf(float x) {
    return neuralseed_fast_tanh(x);
}
