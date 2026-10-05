#include "model_json.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {
struct Span { const char* begin; const char* end; };

const char* whitespace(const char* p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    return p;
}

const char* skip_string(const char* p, const char* end) {
    if (p == end || *p++ != '"') return nullptr;
    while (p < end) {
        if (*p == '\\') {
            if (end - p < 2) return nullptr;
            p += 2;
            continue;
        }
        if (*p++ == '"') return p;
    }
    return nullptr;
}

const char* skip_value(const char* p, const char* end, int depth) {
    p = whitespace(p, end);
    if (p == end || depth > 16) return nullptr;
    if (*p == '"') return skip_string(p, end);
    if (*p == '{' || *p == '[') {
        const char close = *p++ == '{' ? '}' : ']';
        p = whitespace(p, end);
        while (p < end && *p != close) {
            if (*p == ',' || *p == ':') { ++p; continue; }
            p = skip_value(p, end, depth + 1);
            if (!p) return nullptr;
            p = whitespace(p, end);
        }
        return p < end ? p + 1 : nullptr;
    }
    while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') ++p;
    return p;
}

Span field(Span object, const char* name) {
    const char* p = whitespace(object.begin, object.end);
    if (p == object.end || *p++ != '{') return {nullptr, nullptr};
    const size_t length = strlen(name);
    while ((p = whitespace(p, object.end)) < object.end && *p != '}') {
        const char* key = p;
        p = skip_string(p, object.end);
        if (!p) break;
        const bool match = static_cast<size_t>(p - key - 2) == length &&
                           memcmp(key + 1, name, length) == 0;
        p = whitespace(p, object.end);
        if (p == object.end || *p++ != ':') break;
        p = whitespace(p, object.end);
        const char* start = p;
        p = skip_value(p, object.end, 0);
        if (!p) break;
        if (match) return {start, p};
        p = whitespace(p, object.end);
        if (p < object.end && *p == ',') ++p;
        else if (p < object.end && *p != '}') break;
    }
    return {nullptr, nullptr};
}

bool number(const char*& p, const char* end, float& value) {
    p = whitespace(p, end);
    char text[64];
    size_t n = 0;
    while (p < end && n + 1 < sizeof(text) &&
           ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' || *p == 'e' || *p == 'E'))
        text[n++] = *p++;
    if (!n) return false;
    text[n] = 0;
    char* parsed = nullptr;
    value = strtof(text, &parsed);
    return parsed == text + n && isfinite(value);
}

bool vector(Span span, float* out, int count) {
    const char* p = whitespace(span.begin, span.end);
    if (p == span.end || *p++ != '[') return false;
    for (int i = 0; i < count; ++i) {
        if (!number(p, span.end, out[i])) return false;
        p = whitespace(p, span.end);
        if (i + 1 < count) {
            if (p == span.end || *p++ != ',') return false;
        }
    }
    p = whitespace(p, span.end);
    return p + 1 == span.end && *p == ']';
}

bool matrix(Span span, float out[30][10], int rows, int columns) {
    const char* p = whitespace(span.begin, span.end);
    if (p == span.end || *p++ != '[') return false;
    for (int row = 0; row < rows; ++row) {
        p = whitespace(p, span.end);
        const char* next = skip_value(p, span.end, 0);
        if (!next || !vector({p, next}, out[row], columns)) return false;
        p = whitespace(next, span.end);
        if (row + 1 < rows) {
            if (p == span.end || *p++ != ',') return false;
        }
    }
    p = whitespace(p, span.end);
    return p + 1 == span.end && *p == ']';
}

bool integer_field(Span object, const char* key, int& value) {
    Span span = field(object, key);
    if (!span.begin) return false;
    float f;
    const char* p = span.begin;
    if (!number(p, span.end, f) || p != span.end || f < -1000 || f > 1000 ||
        f != static_cast<int>(f)) return false;
    value = static_cast<int>(f);
    return true;
}
} // namespace

bool neuralseed_parse_json(const char* bytes, size_t size, NeuralSeedWeights* out) {
    if (!bytes || !out || !size || size > 128 * 1024) return false;
    Span root{bytes, bytes + size};
    Span info = field(root, "model_data");
    Span state = field(root, "state_dict");
    if (!info.begin || !state.begin) return false;
    NeuralSeedWeights result{};
    if (!integer_field(info, "input_size", result.inputs) ||
        !integer_field(info, "hidden_size", result.hidden) ||
        result.inputs < 1 || result.inputs > 4 || result.hidden < 1 || result.hidden > 10)
        return false;
    Span unit = field(info, "unit_type");
    if (!unit.begin || unit.end - unit.begin != 5 || memcmp(unit.begin, "\"GRU\"", 5) != 0)
        return false;
    int skip = 0, layers = 0, outputs = 0;
    if (!integer_field(info, "skip", skip) || skip != 1 ||
        !integer_field(info, "num_layers", layers) || layers != 1 ||
        !integer_field(info, "output_size", outputs) || outputs != 1)
        return false;

    float wi[30][10]{}, wh[30][10]{}, bi[30]{}, bh[30]{};
    const int gates = 3 * result.hidden;
    if (!matrix(field(state, "rec.weight_ih_l0"), wi, gates, result.inputs) ||
        !matrix(field(state, "rec.weight_hh_l0"), wh, gates, result.hidden) ||
        !vector(field(state, "rec.bias_ih_l0"), bi, gates) ||
        !vector(field(state, "rec.bias_hh_l0"), bh, gates)) return false;
    float dense_matrix[30][10]{};
    if (!matrix(field(state, "lin.weight"), dense_matrix, 1, result.hidden) ||
        !vector(field(state, "lin.bias"), &result.dense_bias, 1)) return false;
    for (int i = 0; i < result.hidden; ++i) result.dense[i] = dense_matrix[0][i];
    // PyTorch stores r,z,n; NeuralSeed's RTNeural setup uses z,r,n.
    for (int gate = 0; gate < 3; ++gate) {
        const int source_gate = gate == 0 ? 1 : gate == 1 ? 0 : 2;
        for (int j = 0; j < result.hidden; ++j) {
            const int source = source_gate * result.hidden + j;
            const int target = gate * result.hidden + j;
            result.bias[0][target] = bi[source];
            result.bias[1][target] = bh[source];
            for (int i = 0; i < result.inputs; ++i) result.wi[i][target] = wi[source][i];
            for (int i = 0; i < result.hidden; ++i) result.wh[i][target] = wh[source][i];
        }
    }
    *out = result;
    return true;
}
