constant uint INVALID_TOKEN = 0xffffffffu;
constant uint RECORD_WIDTH = 3;

kernel void candidates_init(device const float* input [[buffer(0)]],
                            device float* values [[buffer(1)]],
                            device uint* ids [[buffer(2)]], constant uint& width [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    values[index] = input[index];
    ids[index] = index % width;
}

kernel void candidates_merge(device const float* values [[buffer(0)]],
                             device const uint* ids [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             device uint* output_ids [[buffer(3)]], constant uint2& shape [[buffer(4)]],
                             uint index [[thread_position_in_grid]]) {
    uint width = shape.x;
    uint span = shape.y;
    uint row = index / width * width;
    uint col = index % width;
    uint group = col / (span * 2) * (span * 2);
    uint middle = min(group + span, width);
    bool left = col < middle;
    uint begin = left ? middle : group;
    uint end = left ? min(group + span * 2, width) : middle;
    uint low = begin;
    uint high = end;
    float value = values[index];
    uint token = ids[index];
    while (low < high) {
        uint mid = low + (high - low) / 2;
        float other = values[row + mid];
        bool before = isnan(other) != isnan(value) ? isnan(other)
            : other > value || ((other == value || isnan(other)) && ids[row + mid] < token);
        if (before) low = mid + 1;
        else high = mid;
    }
    uint own_rank = col - (left ? group : middle);
    uint dst = row + group + own_rank + low - begin;
    output[dst] = value;
    output_ids[dst] = token;
}

kernel void rng_probe(device const uint4* events [[buffer(0)]],
                      device uint4* output [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    uint4 e = events[row];
    uint2 bits = threefry2x32(e.z, e.w, e.x, e.y);
    output[row] = uint4(bits, as_type<uint>(uniform_from_uint(bits.x)), as_type<uint>(uniform_from_uint(bits.y)));
}

kernel void rng_event(device const uint* state [[buffer(0)]],
                      device uint4* event [[buffer(1)]], constant uint4& key_attempt [[buffer(2)]]) {
    event[0] = uint4(key_attempt.x, key_attempt.y, state[1], key_attempt.z);
}

kernel void rng_select(device const float* weights [[buffer(0)]],
                       device const uint* ids [[buffer(1)]],
                       device const float* reporting [[buffer(2)]],
                       device const uint4* events [[buffer(3)]],
                       device uint* records [[buffer(4)]],
                       constant uint4& shape [[buffer(5)]],
                       constant float2& filters [[buffer(6)]], uint row [[thread_position_in_grid]]) {
    uint width = shape.x;
    uint count = shape.z != 0 || shape.y == 0 ? width : min(width, shape.y);
    uint offset = row * width;
    uint record = row * RECORD_WIDTH;
    records[record] = INVALID_TOKEN;
    records[record + 1] = as_type<uint>(-INFINITY);
    records[record + 2] = 1;
    float mass = 0;
    float max_weight = -INFINITY;
    uint best = 0;
    for (uint i = 0; i < count; ++i) {
        float w = weights[offset + i];
        if (isnan(w) || w == INFINITY || (shape.z == 0 && (!isfinite(w) || w < 0))) return;
        mass += w;
        if (w > max_weight || (w == max_weight && ids[offset + i] < ids[offset + best])) {
            max_weight = w;
            best = i;
        }
    }
    if (shape.z != 0) {
        if (!isfinite(max_weight)) return;
    } else if (!(mass > 0) || !isfinite(mass)) return;
    uint chosen = best;
    if (shape.z == 0) {
        float cutoff = filters.x > 0 && filters.x < 1 ? filters.x * mass : INFINITY;
        float threshold = filters.y > 0 && filters.y < 1 ? filters.y * max_weight : -1;
        float cumulative = 0;
        float retained = 0;
        uint end = 0;
        for (uint i = 0; i < count && cumulative < cutoff; ++i) {
            float w = weights[offset + i];
            cumulative += w;
            if (w > threshold) retained += w;
            end = i + 1;
        }
        if (!(retained > 0)) return;
        uint4 e = events[row];
        float target = uniform_from_uint(threefry2x32(e.z, e.w, e.x, e.y).x) * retained;
        cumulative = 0;
        for (uint i = 0; i < end; ++i) {
            float w = weights[offset + i];
            if (w <= threshold || w <= 0) continue;
            chosen = i;
            cumulative += w;
            if (cumulative > target) break;
        }
    }
    records[record] = ids[offset + chosen];
    records[record + 1] = as_type<uint>(log(reporting[offset + chosen]));
    records[record + 2] = 0;
}

kernel void history_init(device const uint* tokens [[buffer(0)]],
                         device uint* history [[buffer(1)]],
                         device uint* counts [[buffer(2)]],
                         device uint* state [[buffer(3)]], constant uint4& dims [[buffer(4)]]) {
    for (uint i = 0; i < dims.x; ++i) {
        uint t = tokens[i];
        history[i] = t;
        if (t < dims.z) {
            counts[t] += 1;
            if (i >= dims.y) counts[dims.z + t] += 1;
        }
    }
    state[0] = dims.x;
    state[1] = dims.x - dims.y;
    state[2] = 1;
}

kernel void history_commit(device const uint* records [[buffer(0)]],
                           device const uint* stops [[buffer(1)]],
                           device uint* history [[buffer(2)]],
                           device uint* counts [[buffer(3)]],
                           device uint* state [[buffer(4)]], constant uint4& dims [[buffer(5)]]) {
    if (records[2] != 0 || state[2] == 0 || state[0] >= dims.y) return;
    uint token = records[0];
    history[state[0]++] = token;
    state[1] += 1;
    if (token < dims.x) {
        counts[token] += 1;
        counts[dims.x + token] += 1;
    }
    for (uint i = 0; i < dims.z; ++i) if (stops[i] == token) state[2] = 0;
    if (state[0] >= dims.y) state[2] = 0;
}

kernel void history_penalties(device const float* logits [[buffer(0)]],
                              device const uint* counts [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& vocab [[buffer(3)]],
                              constant float4& params [[buffer(4)]], uint token [[thread_position_in_grid]]) {
    float v = logits[token];
    uint generated = counts[vocab + token];
    v -= params.x * float(generated) + (generated > 0 ? params.y : 0);
    if (counts[token] > 0) v = v >= 0 ? v / params.z : v * params.z;
    output[token] = v;
}
