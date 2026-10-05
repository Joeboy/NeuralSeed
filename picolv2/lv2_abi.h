#pragma once

#include <stdint.h>

#define LV2_URID__map "http://lv2plug.in/ns/ext/urid#map"
#define LV2_ATOM__Object "http://lv2plug.in/ns/ext/atom#Object"
#define LV2_ATOM__Path "http://lv2plug.in/ns/ext/atom#Path"
#define LV2_ATOM__URID "http://lv2plug.in/ns/ext/atom#URID"

typedef void* LV2_Handle;
typedef uint32_t LV2_URID;
typedef struct {
    const char* URI;
    void* data;
} LV2_Feature;
typedef struct {
    void* handle;
    LV2_URID (*map)(void*, const char*);
} LV2_URID_Map;
typedef struct {
    uint32_t size;
    uint32_t type;
} LV2_Atom;
typedef struct {
    uint32_t unit;
    uint32_t pad;
} LV2_Atom_Sequence_Body;
typedef struct {
    LV2_Atom atom;
    LV2_Atom_Sequence_Body body;
} LV2_Atom_Sequence;
typedef struct {
    int64_t frames;
    LV2_Atom body;
} LV2_Atom_Event;
typedef struct {
    uint32_t id;
    uint32_t otype;
} LV2_Atom_Object_Body;
typedef struct {
    uint32_t key;
    uint32_t context;
    LV2_Atom value;
} LV2_Atom_Property_Body;
typedef struct LV2_Descriptor {
    const char* URI;
    LV2_Handle (*instantiate)(const struct LV2_Descriptor*, double, const char*, const LV2_Feature* const*);
    void (*connect_port)(LV2_Handle, uint32_t, void*);
    void (*activate)(LV2_Handle);
    void (*run)(LV2_Handle, uint32_t);
    void (*deactivate)(LV2_Handle);
    void (*cleanup)(LV2_Handle);
    const void* (*extension_data)(const char*);
} LV2_Descriptor;
