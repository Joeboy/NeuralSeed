#include "lv2_abi.h"
#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char* uris[8];
static uint32_t map_uri(void*, const char* uri) {
    for (uint32_t i = 1; i < 8; ++i) if (uris[i] && !strcmp(uris[i], uri)) return i;
    for (uint32_t i = 1; i < 8; ++i) if (!uris[i]) { uris[i] = uri; return i; }
    return 0;
}
static size_t padded(size_t n) { return (n + 7) & ~size_t(7); }

int main() {
    void* library = dlopen("build/pc/plugin.so", RTLD_NOW);
    assert(library);
    using DescriptorFn = const LV2_Descriptor* (*)(uint32_t);
    auto descriptor_fn = reinterpret_cast<DescriptorFn>(dlsym(library, "lv2_descriptor"));
    assert(descriptor_fn);
    const LV2_Descriptor* descriptor = descriptor_fn(0);
    assert(descriptor && !descriptor_fn(1));
    LV2_URID_Map map{nullptr, map_uri};
    LV2_Feature feature{LV2_URID__map, &map};
    const LV2_Feature* features[] = {&feature, nullptr};
    LV2_Handle plugin = descriptor->instantiate(descriptor, 48000.0,
        "build/picolv2/pc/neuralseed.lv2/", features);
    assert(plugin);
    float input[64], output[64]{};
    for (float& sample : input) sample = 0.1f;
    descriptor->connect_port(plugin, 0, input);
    descriptor->connect_port(plugin, 1, output);
    alignas(8) uint8_t buffer[1024]{};
    auto* sequence = reinterpret_cast<LV2_Atom_Sequence*>(buffer);
    sequence->atom.size = sizeof(sequence->body);
    descriptor->connect_port(plugin, 2, sequence);
    descriptor->activate(plugin);
    descriptor->run(plugin, 64);
    const float original = output[63];
    assert(isfinite(original) && fabsf(original - 0.1f) > 0.001f);
    LV2_Handle baseline = descriptor->instantiate(descriptor, 48000.0,
        "build/picolv2/pc/neuralseed.lv2/", features);
    assert(baseline);
    float baseline_output[64]{};
    LV2_Atom_Sequence empty{{sizeof(LV2_Atom_Sequence_Body), 0}, {0, 0}};
    descriptor->connect_port(baseline, 0, input);
    descriptor->connect_port(baseline, 1, baseline_output);
    descriptor->connect_port(baseline, 2, &empty);
    descriptor->activate(baseline);
    descriptor->run(baseline, 64);
    descriptor->run(baseline, 64);

    const char* path = "../models/ts9_gru7_gain10_loss003_TEST.json";
    const uint32_t object_id = map_uri(nullptr, LV2_ATOM__Object);
    const uint32_t path_id = map_uri(nullptr, LV2_ATOM__Path);
    const uint32_t urid_id = map_uri(nullptr, LV2_ATOM__URID);
    const uint32_t set_id = map_uri(nullptr, "http://lv2plug.in/ns/ext/patch#Set");
    const uint32_t property_id = map_uri(nullptr, "http://lv2plug.in/ns/ext/patch#property");
    const uint32_t value_id = map_uri(nullptr, "http://lv2plug.in/ns/ext/patch#value");
    const uint32_t model_id = map_uri(nullptr, "https://joebutton.co.uk/lv2/neuralseed#model");
    auto* event = reinterpret_cast<LV2_Atom_Event*>(buffer + sizeof(*sequence));
    event->body.type = object_id;
    auto* object = reinterpret_cast<LV2_Atom_Object_Body*>(event + 1);
    object->otype = set_id;
    auto* prop = reinterpret_cast<LV2_Atom_Property_Body*>(object + 1);
    prop->key = property_id;
    prop->value = {4, urid_id};
    memcpy(prop + 1, &model_id, 4);
    auto* value = reinterpret_cast<LV2_Atom_Property_Body*>(reinterpret_cast<uint8_t*>(prop + 1) + 8);
    value->key = value_id;
    value->value = {static_cast<uint32_t>(strlen(path) + 1), path_id};
    memcpy(value + 1, path, strlen(path) + 1);
    event->body.size = sizeof(*object) + sizeof(*prop) + 8 + sizeof(*value) + padded(strlen(path) + 1);
    sequence->atom.size = sizeof(sequence->body) + sizeof(*event) + event->body.size;
    descriptor->run(plugin, 64);
    assert(isfinite(output[63]) && fabsf(output[63] - baseline_output[63]) > 0.001f);
    descriptor->cleanup(baseline);
    descriptor->cleanup(plugin);
    dlclose(library);
    puts("NeuralSeed LV2 model chooser: OK");
}
