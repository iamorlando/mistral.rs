kernel void argmax_tiles(device const float* logits [[buffer(0)]], device const uint* counts [[buffer(1)]],
                         device LogitTile* tiles [[buffer(2)]], constant uint& width [[buffer(3)]],
                         constant float4& params [[buffer(4)]], uint tile [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup SelectScratch scratch;
    LogitWeights input {logits, counts, width, params.x, params.y, params.z, params.w, 0};
    BestLogit best {-INFINITY, INVALID_TOKEN, 0};
    for (uint j = 0; j < LOGITS_TILE_VALUES; ++j) {
        uint i = tile * LOGITS_TILE + tid + j * SIMD_WIDTH * SELECT_GROUPS;
        float value = i < width ? input.score(i) : -INFINITY;
        best.invalid |= isnan(value) || value == INFINITY;
        if (value > best.value || (value == best.value && i < best.token)) {
            best.value = value;
            best.token = i;
        }
    }
    best = reduce_best(best, SelectThread {tid, lane, group}, scratch);
    if (tid == 0) tiles[tile] = LogitTile {best.value, 0, best.token, best.invalid};
}

kernel void argmax_finish_commit(device const LogitTile* tiles [[buffer(0)]],
                                 device const uint* stops [[buffer(1)]], device uint* records [[buffer(2)]],
                                 device uint* history [[buffer(3)]], device uint* counts [[buffer(4)]],
                                 device uint* state [[buffer(5)]], constant uint4& dims [[buffer(6)]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    uint tile_count = (dims.x + LOGITS_TILE - 1) / LOGITS_TILE;
    BestLogit best {-INFINITY, INVALID_TOKEN, 0};
    for (uint i = tid; i < tile_count; i += SIMD_WIDTH) {
        LogitTile tile = tiles[i];
        best.invalid |= tile.invalid;
        if (tile.maximum > best.value || (tile.maximum == best.value && tile.best < best.token)) {
            best.value = tile.maximum;
            best.token = tile.best;
        }
    }
    float maximum = simd_max(best.value);
    uint winner = simd_min(best.value == maximum ? best.token : INVALID_TOKEN);
    uint invalid = simd_or(best.invalid);
    if (tid == 0) {
        bool valid = invalid == 0 && isfinite(maximum);
        uint token = valid ? winner : INVALID_TOKEN;
        records[0] = token;
        records[1] = as_type<uint>(valid ? 0.0f : -INFINITY);
        records[2] = !valid;
        if (valid) HistoryCommit {stops, history, counts, state, dims}.apply(token);
    }
}
