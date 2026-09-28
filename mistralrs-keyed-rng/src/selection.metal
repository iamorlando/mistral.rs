constant uint INVALID_TOKEN = 0xffffffffu;
constant uint RECORD_WIDTH = 3;
constant uint SIMD_WIDTH = 32;
constant uint SELECT_GROUPS = 8;

struct SelectScratch {
    float mass[SELECT_GROUPS];
    float maximum[SELECT_GROUPS];
    uint best_id[SELECT_GROUPS];
    uint best_index[SELECT_GROUPS];
    uint invalid[SELECT_GROUPS];
    float prefix[SELECT_GROUPS];
    float total;
    float cutoff;
    float threshold;
    float target;
    uint chosen;
    uint selected_group;
    uint valid;
};

struct LogitWeights {
    device const float* logits;
    device const uint* counts;
    uint width;
    float inverse_temperature;
    float frequency;
    float presence;
    float repetition;
    float shift;
    float score(uint i) const {
        float v = logits[i];
        uint generated = counts[width + i];
        v -= frequency * float(generated) + (generated > 0 ? presence : 0);
        if (counts[i] > 0) v = v >= 0 ? v / repetition : v * repetition;
        return v * inverse_temperature;
    }
    float value(uint i) const { return exp(score(i) - shift); }
    uint token(uint i) const { return i; }
};

struct SelectThread {
    uint index;
    uint lane;
    uint group;
};

struct CandidateWeights {
    device const float* weights;
    device const uint* ids;
    float value(uint i) const { return weights[i]; }
    uint token(uint i) const { return ids[i]; }
};

template <typename Weights>
inline uint parallel_select(Weights input, uint count, float draw, bool greedy,
                            float2 filters, SelectThread t, threadgroup SelectScratch& s) {
    uint tiles = (count + SIMD_WIDTH - 1) / SIMD_WIDTH;
    uint span = (tiles + SELECT_GROUPS - 1) / SELECT_GROUPS * SIMD_WIDTH;
    uint begin = t.group * span;
    uint end = min(begin + span, count);
    float mass = 0;
    float best = -INFINITY;
    uint best_id = INVALID_TOKEN;
    uint best_index = INVALID_TOKEN;
    uint invalid = 0;
    for (uint base = begin; base < end; base += SIMD_WIDTH) {
        uint i = base + t.lane;
        if (i < end) {
            float w = input.value(i);
            uint token = input.token(i);
            invalid |= isnan(w) || w == INFINITY || (!greedy && (!isfinite(w) || w < 0));
            mass += w;
            if (w > best || (w == best && token < best_id)) {
                best = w;
                best_id = token;
                best_index = i;
            }
        }
    }
    float group_best = simd_max(best);
    uint group_id = simd_min(best == group_best ? best_id : INVALID_TOKEN);
    uint group_index = simd_min(best == group_best && best_id == group_id ? best_index : INVALID_TOKEN);
    float group_mass = simd_sum(mass);
    uint group_invalid = simd_or(invalid);
    if (t.lane == 0) {
        s.mass[t.group] = group_mass;
        s.maximum[t.group] = group_best;
        s.best_id[t.group] = group_id;
        s.best_index[t.group] = group_index;
        s.invalid[t.group] = group_invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t.index == 0) {
        s.total = 0;
        s.valid = 1;
        s.chosen = INVALID_TOKEN;
        best = -INFINITY;
        best_id = INVALID_TOKEN;
        for (uint g = 0; g < SELECT_GROUPS; ++g) {
            s.prefix[g] = s.total;
            s.total += s.mass[g];
            s.valid &= s.invalid[g] == 0;
            if (s.maximum[g] > best || (s.maximum[g] == best && s.best_id[g] < best_id)) {
                best = s.maximum[g];
                best_id = s.best_id[g];
                s.chosen = s.best_index[g];
            }
        }
        s.valid &= greedy ? isfinite(best) : s.total > 0 && isfinite(s.total);
        s.cutoff = filters.x > 0 && filters.x < 1 ? filters.x * s.total : INFINITY;
        s.threshold = filters.y > 0 && filters.y < 1 ? filters.y * best : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!s.valid) return INVALID_TOKEN;
    if (greedy) return s.chosen;
    bool filtered = isfinite(s.cutoff) || s.threshold >= 0;
    if (filtered) {
        float before = s.prefix[t.group];
        mass = 0;
        for (uint base = begin; base < end; base += SIMD_WIDTH) {
            uint i = base + t.lane;
            float w = i < end ? input.value(i) : 0;
            float prefix = simd_prefix_exclusive_sum(w);
            if (before + prefix < s.cutoff && w > s.threshold) mass += w;
            before += simd_sum(w);
        }
        group_mass = simd_sum(mass);
        if (t.lane == 0) s.mass[t.group] = group_mass;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t.index == 0) {
        float retained = 0;
        for (uint g = 0; g < SELECT_GROUPS; ++g) retained += s.mass[g];
        s.valid = retained > 0 && isfinite(retained);
        s.target = draw * retained;
        s.selected_group = 0;
        float before = 0;
        float selected_before = 0;
        for (uint g = 0; g < SELECT_GROUPS; ++g) {
            if (s.mass[g] > 0) {
                s.selected_group = g;
                selected_before = before;
            }
            if (before + s.mass[g] > s.target) break;
            before += s.mass[g];
        }
        s.target -= selected_before;
        s.chosen = INVALID_TOKEN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!s.valid) return INVALID_TOKEN;
    if (t.group == s.selected_group) {
        float raw_before = s.prefix[t.group];
        float before = 0;
        uint last = INVALID_TOKEN;
        for (uint base = begin; base < end; base += SIMD_WIDTH) {
            uint i = base + t.lane;
            float w = i < end ? input.value(i) : 0;
            float raw_prefix = simd_prefix_exclusive_sum(w);
            float kept = raw_before + raw_prefix < s.cutoff && w > s.threshold ? w : 0;
            float cumulative = before + simd_prefix_inclusive_sum(kept);
            uint winner = simd_min(kept > 0 && cumulative > s.target ? i : INVALID_TOKEN);
            uint tail = simd_max(kept > 0 ? i + 1 : 0u);
            if (tail > 0) last = tail - 1;
            if (winner != INVALID_TOKEN) {
                if (t.lane == 0) s.chosen = winner;
                break;
            }
            before += simd_sum(kept);
            raw_before += simd_sum(w);
        }
        if (t.lane == 0 && s.chosen == INVALID_TOKEN) s.chosen = last;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s.chosen;
}

kernel void rng_logits(device const float* logits [[buffer(0)]],
                       device const uint* counts [[buffer(1)]], device const uint* state [[buffer(2)]],
                       device uint* records [[buffer(3)]], device uint* tokens [[buffer(4)]],
                       constant uint* shape [[buffer(5)]], constant float* params [[buffer(6)]],
                       uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                       uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup SelectScratch scratch;
    threadgroup float maximum;
    threadgroup uint best_index;
    threadgroup uint valid;
    uint width = shape[0];
    LogitWeights weights {logits, counts, width, params[0], params[1], params[2], params[3], 0};
    float best = -INFINITY;
    uint best_id = INVALID_TOKEN;
    uint invalid = 0;
    for (uint i = tid; i < width; i += SIMD_WIDTH * SELECT_GROUPS) {
        float v = weights.score(i);
        invalid |= isnan(v) || v == INFINITY;
        if (v > best || (v == best && i < best_id)) {
            best = v;
            best_id = i;
        }
    }
    float group_best = simd_max(best);
    uint group_id = simd_min(best == group_best ? best_id : INVALID_TOKEN);
    uint group_invalid = simd_or(invalid);
    if (lane == 0) {
        scratch.maximum[group] = group_best;
        scratch.best_id[group] = group_id;
        scratch.invalid[group] = group_invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        maximum = -INFINITY;
        best_index = INVALID_TOKEN;
        valid = 1;
        for (uint g = 0; g < SELECT_GROUPS; ++g) {
            valid &= scratch.invalid[g] == 0;
            if (scratch.maximum[g] > maximum || (scratch.maximum[g] == maximum && scratch.best_id[g] < best_index)) {
                maximum = scratch.maximum[g];
                best_index = scratch.best_id[g];
            }
        }
        valid &= isfinite(maximum);
        records[0] = INVALID_TOKEN;
        records[1] = as_type<uint>(-INFINITY);
        records[2] = 1;
        tokens[0] = INVALID_TOKEN;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid) return;
    weights.shift = maximum;
    uint chosen;
    if (shape[1] != 0) {
        float mass = 0;
        for (uint i = tid; i < width; i += SIMD_WIDTH * SELECT_GROUPS) mass += weights.value(i);
        float group_mass = simd_sum(mass);
        if (lane == 0) scratch.mass[group] = group_mass;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            scratch.total = 0;
            for (uint g = 0; g < SELECT_GROUPS; ++g) scratch.total += scratch.mass[g];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        chosen = best_index;
    } else {
        float draw = uniform_from_uint(threefry2x32(state[1], shape[4], shape[2], shape[3]).x);
        chosen = parallel_select(weights, width, draw,
                                 false, float2(1, params[4]), SelectThread {tid, lane, group}, scratch);
    }
    if (tid == 0 && chosen != INVALID_TOKEN) {
        records[0] = chosen;
        records[1] = as_type<uint>(weights.score(chosen) - maximum - log(scratch.total));
        records[2] = 0;
        tokens[0] = chosen;
    }
}

kernel void rng_select(device const float* weights [[buffer(0)]],
                       device const uint* ids [[buffer(1)]],
                       device const float* reporting [[buffer(2)]],
                       device const uint4* events [[buffer(3)]],
                       device uint* records [[buffer(4)]], device uint* tokens [[buffer(5)]],
                       constant uint4& shape [[buffer(6)]], constant float2& filters [[buffer(7)]],
                       uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup SelectScratch scratch;
    uint width = shape.x;
    uint count = shape.z != 0 || shape.y == 0 ? width : min(width, shape.y);
    uint offset = row * width;
    uint4 event = events[row];
    float draw = uniform_from_uint(threefry2x32(event.z, event.w, event.x, event.y).x);
    uint chosen = parallel_select(CandidateWeights {weights + offset, ids + offset}, count,
                                  draw, shape.z != 0, filters, SelectThread {tid, lane, group}, scratch);
    if (tid == 0) {
        uint token = chosen == INVALID_TOKEN ? INVALID_TOKEN : ids[offset + chosen];
        records[row * RECORD_WIDTH] = token;
        records[row * RECORD_WIDTH + 1] = as_type<uint>(chosen == INVALID_TOKEN ? -INFINITY : log(reporting[offset + chosen]));
        records[row * RECORD_WIDTH + 2] = chosen == INVALID_TOKEN;
        tokens[row] = token;
    }
}

constant uint LOGITS_TILE = 1024;
constant uint LOGITS_TILE_VALUES = LOGITS_TILE / (SIMD_WIDTH * SELECT_GROUPS);

struct LogitTile {
    float maximum;
    float mass;
    uint best;
    uint invalid;
};

struct TileWeights {
    device const float* weights;
    float value(uint i) const { return weights[i]; }
    uint token(uint i) const { return i; }
};

kernel void logits_tiles(device const float* logits [[buffer(0)]], device const uint* counts [[buffer(1)]],
                         device float* weights [[buffer(2)]], device LogitTile* tiles [[buffer(3)]],
                         constant uint& width [[buffer(4)]], constant float4* all_params [[buffer(5)]],
                         uint batch_tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup SelectScratch scratch;
    threadgroup float maximum;
    uint tile_count = (width + LOGITS_TILE - 1) / LOGITS_TILE;
    uint row = batch_tile / tile_count;
    uint tile = batch_tile % tile_count;
    logits += row * width;
    counts += row * 2 * width;
    weights += row * width;
    tiles += row * tile_count;
    float4 params = all_params[row];
    LogitWeights input {logits, counts, width, params.x, params.y, params.z, params.w, 0};
    float values[LOGITS_TILE_VALUES];
    float best = -INFINITY;
    uint best_id = INVALID_TOKEN;
    uint invalid = 0;
    for (uint j = 0; j < LOGITS_TILE_VALUES; ++j) {
        uint i = tile * LOGITS_TILE + tid + j * SIMD_WIDTH * SELECT_GROUPS;
        float v = i < width ? input.score(i) : -INFINITY;
        values[j] = v;
        invalid |= isnan(v) || v == INFINITY;
        if (v > best || (v == best && i < best_id)) { best = v; best_id = i; }
    }
    float group_best = simd_max(best);
    uint group_id = simd_min(best == group_best ? best_id : INVALID_TOKEN);
    uint group_invalid = simd_or(invalid);
    if (lane == 0) {
        scratch.maximum[group] = group_best;
        scratch.best_id[group] = group_id;
        scratch.invalid[group] = group_invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        maximum = -INFINITY;
        best_id = INVALID_TOKEN;
        invalid = 0;
        for (uint g = 0; g < SELECT_GROUPS; ++g) {
            invalid |= scratch.invalid[g];
            if (scratch.maximum[g] > maximum || (scratch.maximum[g] == maximum && scratch.best_id[g] < best_id)) {
                maximum = scratch.maximum[g];
                best_id = scratch.best_id[g];
            }
        }
        tiles[tile].maximum = maximum;
        tiles[tile].best = best_id;
        tiles[tile].invalid = invalid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mass = 0;
    for (uint j = 0; j < LOGITS_TILE_VALUES; ++j) {
        uint i = tile * LOGITS_TILE + tid + j * SIMD_WIDTH * SELECT_GROUPS;
        float w = isfinite(maximum) ? exp(values[j] - maximum) : 0;
        mass += w;
        if (i < width) weights[i] = w;
    }
    float group_mass = simd_sum(mass);
    if (lane == 0) scratch.mass[group] = group_mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        mass = 0;
        for (uint g = 0; g < SELECT_GROUPS; ++g) mass += scratch.mass[g];
        tiles[tile].mass = mass;
    }
}

kernel void logits_finish(device const float* weights [[buffer(0)]], device const LogitTile* tiles [[buffer(1)]],
                          device const uint* state [[buffer(2)]], device uint* records [[buffer(3)]],
                          device uint* tokens [[buffer(4)]], constant uint* shape [[buffer(5)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup SelectScratch scratch;
    threadgroup float maximum;
    threadgroup float total;
    threadgroup float conditional;
    threadgroup uint chosen_tile;
    threadgroup uint best_index;
    threadgroup uint valid;
    shape += row * 5;
    uint width = shape[0];
    uint count = (width + LOGITS_TILE - 1) / LOGITS_TILE;
    weights += row * width;
    tiles += row * count;
    state += row * 3;
    records += row * RECORD_WIDTH;
    tokens += row;
    if (tid == 0) {
        maximum = -INFINITY;
        best_index = INVALID_TOKEN;
        valid = 1;
        for (uint i = 0; i < count; ++i) {
            valid &= tiles[i].invalid == 0;
            if (tiles[i].maximum > maximum || (tiles[i].maximum == maximum && tiles[i].best < best_index)) {
                maximum = tiles[i].maximum;
                best_index = tiles[i].best;
            }
        }
        valid &= isfinite(maximum);
        records[0] = INVALID_TOKEN;
        records[1] = as_type<uint>(-INFINITY);
        records[2] = 1;
        tokens[0] = INVALID_TOKEN;
        total = 0;
        for (uint i = 0; i < count; ++i) total += tiles[i].mass * exp(tiles[i].maximum - maximum);
        if (shape[1] == 0) {
            float draw = uniform_from_uint(threefry2x32(state[1], shape[4], shape[2], shape[3]).x);
            float target = draw * total;
            float before = 0;
            for (uint i = 0; i < count; ++i) {
                float mass = tiles[i].mass * exp(tiles[i].maximum - maximum);
                if (mass > 0) { chosen_tile = i; conditional = (target - before) / mass; }
                if (before + mass > target) break;
                before += mass;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!valid) return;
    uint chosen = best_index;
    if (shape[1] == 0) {
        uint offset = chosen_tile * LOGITS_TILE;
        chosen = parallel_select(TileWeights {weights + offset}, min(width - offset, LOGITS_TILE),
                                 conditional, false, float2(1, 0), SelectThread {tid, lane, group}, scratch);
        if (chosen != INVALID_TOKEN) chosen += offset;
    }
    if (tid == 0 && chosen != INVALID_TOKEN) {
        records[0] = chosen;
        records[1] = as_type<uint>(log(weights[chosen]) + tiles[chosen / LOGITS_TILE].maximum - maximum - log(total));
        records[2] = 0;
        tokens[0] = chosen;
    }
}

kernel void batch_logits(device const float* logits [[buffer(0)]], device const uint* counts [[buffer(1)]],
                         device const uint* states [[buffer(2)]], device float* scores [[buffer(3)]],
                         device uint4* events [[buffer(4)]], constant uint4* config [[buffer(5)]],
                         constant float4* params [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    uint width = config[0].x;
    uint row = index / width;
    uint token = index % width;
    float4 p = params[row];
    LogitWeights input {logits + row * width, counts + row * 2 * width, width, p.x, p.y, p.z, p.w, 0};
    scores[index] = input.score(token);
    if (token == 0) events[row] = uint4(config[row].y, config[row].z, states[row * 3 + 1], config[row].w);
}
