#include "fast_math.h"
#include "lv2_abi.h"
#include <assert.h>
#include <dlfcn.h>
#include <initializer_list>
#include <math.h>
#include <stdio.h>
#include <string.h>

static const char* uris[8];
static uint32_t map_uri(void*, const char* uri) {
    for (uint32_t i = 1; i < 8; ++i) if (uris[i] && !strcmp(uris[i], uri)) return i;
    for (uint32_t i = 1; i < 8; ++i) if (!uris[i]) { uris[i] = uri; return i; }
    return 0;
}

struct Plugin {
    void* library;
    const LV2_Descriptor* descriptor;
    LV2_Handle handle;
    float input[512]{};
    float output[512]{};
    LV2_Atom_Sequence empty{{sizeof(LV2_Atom_Sequence_Body), 0}, {0, 0}};
};

static Plugin open_plugin(const char* binary, const char* bundle, const LV2_Feature* const* features) {
    Plugin plugin{};
    plugin.library = dlopen(binary, RTLD_NOW);
    assert(plugin.library);
    using DescriptorFn = const LV2_Descriptor* (*)(uint32_t);
    plugin.descriptor = reinterpret_cast<DescriptorFn>(dlsym(plugin.library, "lv2_descriptor"))(0);
    plugin.handle = plugin.descriptor->instantiate(plugin.descriptor, 48000.0, bundle, features);
    assert(plugin.handle);
    // The caller connects ports after the returned struct reaches its final address.
    return plugin;
}

int main() {
    float max_tanh_error = 0;
    for (int i = -8000; i <= 8000; ++i) {
        const float x = i / 1000.0f;
        max_tanh_error = fmaxf(max_tanh_error, fabsf(neuralseed_fast_tanh(x) - tanhf(x)));
    }
    assert(max_tanh_error < 0.00004f);
    LV2_URID_Map map{nullptr, map_uri};
    LV2_Feature feature{LV2_URID__map, &map};
    const LV2_Feature* features[] = {&feature, nullptr};
    Plugin exact = open_plugin("build/pc/plugin.so", "build/picolv2/pc/neuralseed.lv2/", features);
    Plugin fast = open_plugin("build/pcfast/plugin.so", "build/picolv2/pcfast/neuralseed.lv2/", features);
    Plugin dynamic = open_plugin("build/pcdynamic/plugin.so", "build/picolv2/pcdynamic/neuralseed.lv2/", features);
    for (Plugin* plugin : {&exact, &fast, &dynamic}) {
        plugin->descriptor->connect_port(plugin->handle, 0, plugin->input);
        plugin->descriptor->connect_port(plugin->handle, 1, plugin->output);
        plugin->descriptor->connect_port(plugin->handle, 2, &plugin->empty);
        plugin->descriptor->activate(plugin->handle);
    }
    float max_audio_error = 0, max_layer_error = 0;
    for (int block = 0; block < 16; ++block) {
        for (int i = 0; i < 512; ++i) {
            const float sample = 0.2f * sinf((block * 512 + i) * 0.057f);
            exact.input[i] = fast.input[i] = dynamic.input[i] = sample;
        }
        exact.descriptor->run(exact.handle, 512);
        fast.descriptor->run(fast.handle, 512);
        dynamic.descriptor->run(dynamic.handle, 512);
        for (int i = 0; i < 512; ++i) {
            max_audio_error = fmaxf(max_audio_error, fabsf(exact.output[i] - fast.output[i]));
            max_layer_error = fmaxf(max_layer_error, fabsf(dynamic.output[i] - fast.output[i]));
        }
    }
    assert(max_audio_error < 0.001f);
    assert(max_layer_error < 0.001f);
    printf("fast activations: max tanh error %.8g, max audio error %.8g, max static/dynamic error %.8g\n",
           max_tanh_error, max_audio_error, max_layer_error);
    for (Plugin* plugin : {&exact, &fast, &dynamic}) {
        plugin->descriptor->cleanup(plugin->handle);
        dlclose(plugin->library);
    }
}
