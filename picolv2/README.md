# NeuralSeed for PicoLV2

This is an LV2 adapter for the [upstream NeuralSeed](https://github.com/GuitarML/NeuralSeed)
checkout. It uses the checkout's pinned RTNeural GRU/Dense implementation and
DaisySP state variable filter, with the same dry skip connection, constant
power mix, and four optional EQ boosts as the pedal code. It runs mono at 48 kHz.

The file chooser accepts the PyTorch training JSON format used by NeuralSeed:
one GRU layer, one dense output, a dry skip connection, 1–4 inputs, and up to
10 hidden units. The first input is audio; later inputs are the three parameter
controls. The bundle ships the upstream `gru8_ts9_pytorch.json` model as its
default. The 11 models compiled into the Daisy firmware are not included.
Choosing an invalid or unavailable file leaves the previous model active.

Build from this directory:

```sh
make TARGET=pico bundle
make TARGET=pc bundle
make test
make test-fast
```

The bundles are written to `build/picolv2/{pico,pc}/neuralseed.lv2`.
The PC build uses RTNeural's original libm activations. The Pico build replaces
the GRU sigmoid and tanh operations with a small interpolated tanh table.
Models with one input and eight hidden units use RTNeural's fixed-size GRU and
dense layers on both builds; other supported JSON models use RTNeural's dynamic
layers. `make test-fast`
compares the Pico activation path against the PC reference on a host build.
`picolv2-resources.txt` includes the default JSON at `/neuralseed/model.json`.
The PicoLV2 file chooser sends a `patch:Set` with an `atom:Path` to select a
different JSON file. Live changes parse and install a model during the LV2
`run` call, so switching models can interrupt one audio block. Steady-state
audio processing does not allocate memory.

The Pico build uses the PicoLV2 plugin SDK at
`../../../../picolv2/plugin-src/sdk` by default. Override `SDK_ROOT` for
another location. The RTNeural and DaisySP
submodules must be initialized with `git submodule update --init RTNeural DaisySP`.
