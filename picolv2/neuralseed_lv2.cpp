// LV2 adapter for the original RTNeural and DaisySP processing used by NeuralSeed.
#include "model_json.h"

#include "lv2_abi.h"
#include <picolv2/filesystem.h>
#include <array>

#ifdef NEURALSEED_FAST_ACTIVATIONS
#include "fast_math.h"
#include <RTNeural/common.h>
namespace RTNeural {
template <> inline float sigmoid<float>(float value) noexcept {
    return 0.5f * (1.0f + neuralseed_fast_tanh(0.5f * value));
}
}
#endif

#include <RTNeural/gru/gru.h>
#include <RTNeural/gru/gru.tpp>
#include <RTNeural/dense/dense.h>
#include <Filters/svf.h>

#include <math.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define PLUGIN_URI "https://joebutton.co.uk/lv2/neuralseed"
#define MODEL_URI PLUGIN_URI "#model"

namespace {
struct Instance {
    const float* input = nullptr;
    float* output = nullptr;
    const LV2_Atom_Sequence* control = nullptr;
    const float* knobs[11]{};
    const PicoLV2_Filesystem* filesystem = nullptr;
    RTNeural::GRULayer<float>* gru = nullptr;
    RTNeural::Dense<float>* dense = nullptr;
    RTNeural::GRULayerT<float, 1, 8>* gru8 = nullptr;
    RTNeural::DenseT<float, 8, 1>* dense8 = nullptr;
    daisysp::Svf filters[4];
    int model_inputs = 0;
    LV2_URID object = 0, path = 0, urid = 0, set = 0, property = 0, value = 0, model = 0;
};

float knob(const float* value, float fallback, float low, float high) {
    const float v = value ? *value : fallback;
    return isfinite(v) ? fminf(high, fmaxf(low, v)) : fallback;
}

bool install_model(Instance* instance, const char* data, size_t size) {
    NeuralSeedWeights weights{};
    if (!neuralseed_parse_json(data, size, &weights)) return false;
#ifndef NEURALSEED_FORCE_DYNAMIC
    if (weights.inputs == 1 && weights.hidden == 8) {
        auto* gru = new (std::nothrow) RTNeural::GRULayerT<float, 1, 8>;
        auto* dense = new (std::nothrow) RTNeural::DenseT<float, 8, 1>;
        if (!gru || !dense) { delete gru; delete dense; return false; }
        std::vector<std::vector<float>> wi(1, std::vector<float>(24));
        std::vector<std::vector<float>> wh(8, std::vector<float>(24));
        std::vector<std::vector<float>> bias(2, std::vector<float>(24));
        for (int j = 0; j < 24; ++j) {
            wi[0][j] = weights.wi[0][j];
            for (int i = 0; i < 8; ++i) wh[i][j] = weights.wh[i][j];
            bias[0][j] = weights.bias[0][j];
            bias[1][j] = weights.bias[1][j];
        }
        gru->setWVals(wi);
        gru->setUVals(wh);
        gru->setBVals(bias);
        float* linear[1] = {weights.dense};
        dense->setWeights(linear);
        dense->setBias(&weights.dense_bias);
        gru->reset();
        delete instance->gru;
        delete instance->dense;
        delete instance->gru8;
        delete instance->dense8;
        instance->gru = nullptr;
        instance->dense = nullptr;
        instance->gru8 = gru;
        instance->dense8 = dense;
        instance->model_inputs = 1;
        return true;
    }
#endif
    auto* gru = new (std::nothrow) RTNeural::GRULayer<float>(weights.inputs, weights.hidden);
    auto* dense = new (std::nothrow) RTNeural::Dense<float>(weights.hidden, 1);
    if (!gru || !dense) { delete gru; delete dense; return false; }
    float* wi[4], *wh[10], *bias[2], *linear[1];
    for (int i = 0; i < weights.inputs; ++i) wi[i] = weights.wi[i];
    for (int i = 0; i < weights.hidden; ++i) wh[i] = weights.wh[i];
    bias[0] = weights.bias[0]; bias[1] = weights.bias[1];
    linear[0] = weights.dense;
    gru->setWVals(wi);
    gru->setUVals(wh);
    gru->setBVals(bias);
    dense->setWeights(linear);
    dense->setBias(&weights.dense_bias);
    gru->reset();
    delete instance->gru;
    delete instance->dense;
    delete instance->gru8;
    delete instance->dense8;
    instance->gru = gru;
    instance->dense = dense;
    instance->gru8 = nullptr;
    instance->dense8 = nullptr;
    instance->model_inputs = weights.inputs;
    return true;
}

bool load_path(Instance* instance, const char* path) {
    if (!path) return false;
    if (instance->filesystem) {
        uint32_t size = 0;
        const uint8_t* bytes = instance->filesystem->map(instance->filesystem->handle, path, &size);
        return bytes && install_model(instance, reinterpret_cast<const char*>(bytes), size);
    }
#ifndef PICOLV2_TARGET
    FILE* file = fopen(path, "rb");
    if (!file) return false;
    bool success = false;
    if (fseek(file, 0, SEEK_END) == 0) {
        const long size = ftell(file);
        if (size > 0 && size <= 128 * 1024 && fseek(file, 0, SEEK_SET) == 0) {
            char* bytes = static_cast<char*>(malloc(size));
            if (bytes) {
                if (fread(bytes, 1, size, file) == static_cast<size_t>(size))
                    success = install_model(instance, bytes, size);
                free(bytes);
            }
        }
    }
    fclose(file);
    return success;
#else
    return false;
#endif
}

// Bounded Atom parsing: a file chooser delivers patch:Set(model, atom:Path).
const char* selected_path(const Instance* instance) {
    const auto* sequence = instance->control;
    if (!sequence || sequence->atom.size < sizeof(LV2_Atom_Sequence_Body) ||
        sequence->atom.size > 16 * 1024) return nullptr;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&sequence->body) + sizeof(sequence->body);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(&sequence->body) + sequence->atom.size;
    while (p + sizeof(LV2_Atom_Event) <= end) {
        const auto* event = reinterpret_cast<const LV2_Atom_Event*>(p);
        if (event->body.size > static_cast<size_t>(end - p - sizeof(LV2_Atom_Event))) break;
        const size_t bytes = sizeof(LV2_Atom_Event) + ((event->body.size + 7u) & ~7u);
        if (bytes > static_cast<size_t>(end - p)) break;
        if (event->body.type == instance->object && event->body.size >= sizeof(LV2_Atom_Object_Body)) {
            const auto* object = reinterpret_cast<const LV2_Atom_Object_Body*>(p + sizeof(LV2_Atom_Event));
            if (object->otype == instance->set) {
                const uint8_t* q = reinterpret_cast<const uint8_t*>(object) + sizeof(*object);
                const uint8_t* object_end = p + sizeof(LV2_Atom_Event) + event->body.size;
                LV2_URID property = 0;
                const char* path = nullptr;
                while (q + sizeof(LV2_Atom_Property_Body) <= object_end) {
                    const auto* prop = reinterpret_cast<const LV2_Atom_Property_Body*>(q);
                    if (prop->value.size > static_cast<size_t>(object_end - q - sizeof(LV2_Atom_Property_Body))) break;
                    const size_t length = sizeof(LV2_Atom_Property_Body) + ((prop->value.size + 7u) & ~7u);
                    if (length > static_cast<size_t>(object_end - q)) break;
                    const uint8_t* body = q + sizeof(*prop);
                    if (prop->key == instance->property && prop->value.type == instance->urid && prop->value.size == 4)
                        memcpy(&property, body, 4);
                    if (prop->key == instance->value && prop->value.type == instance->path &&
                        prop->value.size > 1 && prop->value.size <= 4096 &&
                        body[prop->value.size - 1] == 0 &&
                        !memchr(body, 0, prop->value.size - 1))
                        path = reinterpret_cast<const char*>(body);
                    q += length;
                }
                if (property == instance->model) return path;
            }
        }
        p += bytes;
    }
    return nullptr;
}

LV2_Handle instantiate(const LV2_Descriptor*, double rate, const char* bundle,
                       const LV2_Feature* const* features) {
    if (rate != 48000.0) return nullptr;
    const LV2_URID_Map* map = nullptr;
    const PicoLV2_Filesystem* filesystem = nullptr;
    if (features) for (auto feature = features; *feature; ++feature) {
        if (!strcmp((*feature)->URI, LV2_URID__map)) map = static_cast<const LV2_URID_Map*>((*feature)->data);
        if (!strcmp((*feature)->URI, PICOLV2_FILESYSTEM_URI)) filesystem = static_cast<const PicoLV2_Filesystem*>((*feature)->data);
    }
    if (!map) return nullptr;
    auto* instance = new (std::nothrow) Instance;
    if (!instance) return nullptr;
    instance->filesystem = filesystem;
    instance->object = map->map(map->handle, LV2_ATOM__Object);
    instance->path = map->map(map->handle, LV2_ATOM__Path);
    instance->urid = map->map(map->handle, LV2_ATOM__URID);
    instance->set = map->map(map->handle, "http://lv2plug.in/ns/ext/patch#Set");
    instance->property = map->map(map->handle, "http://lv2plug.in/ns/ext/patch#property");
    instance->value = map->map(map->handle, "http://lv2plug.in/ns/ext/patch#value");
    instance->model = map->map(map->handle, MODEL_URI);
    const float frequencies[4] = {120, 400, 800, 1600};
    for (int i = 0; i < 4; ++i) {
        instance->filters[i].Init(48000.0f);
        instance->filters[i].SetRes(0.6f);
        instance->filters[i].SetDrive(0.0f);
        instance->filters[i].SetFreq(frequencies[i]);
    }
    if (filesystem) load_path(instance, "/neuralseed/model.json");
#ifndef PICOLV2_TARGET
    else if (bundle) {
        char path[4096];
        const int length = snprintf(path, sizeof(path), "%s%smodel.json", bundle,
                                    bundle[0] && bundle[strlen(bundle) - 1] == '/' ? "" : "/");
        if (length > 0 && length < static_cast<int>(sizeof(path))) load_path(instance, path);
    }
#endif
    return instance;
}

void connect_port(LV2_Handle handle, uint32_t port, void* data) {
    auto* instance = static_cast<Instance*>(handle);
    if (port == 0) instance->input = static_cast<const float*>(data);
    else if (port == 1) instance->output = static_cast<float*>(data);
    else if (port == 2) instance->control = static_cast<const LV2_Atom_Sequence*>(data);
    else if (port >= 3 && port <= 13) instance->knobs[port - 3] = static_cast<const float*>(data);
}

void activate(LV2_Handle handle) {
    auto* instance = static_cast<Instance*>(handle);
    if (instance->gru) instance->gru->reset();
    if (instance->gru8) instance->gru8->reset();
}

void run(LV2_Handle handle, uint32_t count) {
    auto* instance = static_cast<Instance*>(handle);
    if (const char* path = selected_path(instance)) load_path(instance, path);
    if (!instance->input || !instance->output) return;
    const float input_level = knob(instance->knobs[0], 1, 0, 3);
    const float mix = knob(instance->knobs[1], 1, 0, 1);
    const float output_level = knob(instance->knobs[2], 1, 0, 1);
    const float param1 = knob(instance->knobs[3], .5f, 0, 1);
    const float param2 = knob(instance->knobs[4], .5f, 0, 1);
    const float param3 = knob(instance->knobs[5], .5f, 0, 1);
    const bool bypass = knob(instance->knobs[6], 0, 0, 1) >= .5f;
    bool eq[4];
    for (int i = 0; i < 4; ++i) eq[i] = knob(instance->knobs[7 + i], 0, 0, 1) >= .5f;
    const float dry_gain = cosf(mix * 1.57079632679f);
    const float wet_gain = sinf(mix * 1.57079632679f);
    if (bypass || (!instance->gru && !instance->gru8)) {
        for (uint32_t i = 0; i < count; ++i)
            instance->output[i] = isfinite(instance->input[i]) ? instance->input[i] : 0.0f;
        return;
    }
    if (instance->gru8) {
        auto* gru = instance->gru8;
        auto* dense = instance->dense8;
        for (uint32_t i = 0; i < count; ++i) {
            const float dry = isfinite(instance->input[i]) ? instance->input[i] : 0.0f;
            const float input8[1] = {dry * input_level};
            gru->forward(input8);
            dense->forward(gru->outs);
            float wet = dense->outs[0] + dry;
            float boost = 0;
            for (int band = 0; band < 4; ++band) if (eq[band]) {
                instance->filters[band].Process(wet);
                boost += instance->filters[band].Band() * .5f;
            }
            wet += boost * .7f;
            const float result = (dry * dry_gain + wet * wet_gain) * output_level;
            instance->output[i] = isfinite(result) ? result : 0.0f;
        }
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const float dry = isfinite(instance->input[i]) ? instance->input[i] : 0.0f;
        float inputs[4] = {dry * input_level, param1, param2, param3};
        float hidden[10], neural = 0;
        instance->gru->forward(inputs, hidden);
        instance->dense->forward(hidden, &neural);
        float wet = neural + dry; // NeuralSeed's skip connection
        float boost = 0;
        for (int band = 0; band < 4; ++band) if (eq[band]) {
            instance->filters[band].Process(wet);
            boost += instance->filters[band].Band() * .5f;
        }
        wet += boost * .7f;
        const float result = (dry * dry_gain + wet * wet_gain) * output_level;
        instance->output[i] = isfinite(result) ? result : 0.0f;
    }
}

void deactivate(LV2_Handle) {}
void cleanup(LV2_Handle handle) {
    auto* instance = static_cast<Instance*>(handle);
    delete instance->gru;
    delete instance->dense;
    delete instance->gru8;
    delete instance->dense8;
    delete instance;
}
const void* extension_data(const char*) { return nullptr; }
const LV2_Descriptor descriptor = {
    PLUGIN_URI, instantiate, connect_port, activate, run, deactivate, cleanup, extension_data
};
} // namespace

extern "C" __attribute__((visibility("default")))
const LV2_Descriptor* lv2_descriptor(uint32_t index) { return index == 0 ? &descriptor : nullptr; }
