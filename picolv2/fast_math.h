#pragma once

// Pico-only activation approximation. The table covers |x| <= 8 in steps of
// 1/64; outside that range tanh is saturated. No per-sample libm calls.
float neuralseed_fast_tanh(float x) noexcept;
