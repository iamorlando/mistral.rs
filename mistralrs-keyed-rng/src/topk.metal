constant uint TOPK_BLOCK = 1024;
constant uint TOPK_THREADS = 256;
constant uint TOPK_VALUES = TOPK_BLOCK / TOPK_THREADS;

inline bool candidate_before(float a, uint ai, float b, uint bi) {
    return isnan(a) != isnan(b) ? isnan(a)
        : a > b || ((a == b || isnan(a)) && ai < bi);
}

kernel void candidates_topk_blocks(device const float* input [[buffer(0)]],
                                  device float* output [[buffer(1)]], device uint* output_ids [[buffer(2)]],
                                  constant uint4& shape [[buffer(3)]], uint block [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[TOPK_BLOCK];
    threadgroup uint ids[TOPK_BLOCK];
    float local_values[TOPK_VALUES];
    uint local_ids[TOPK_VALUES];
    uint row = block / shape.z;
    uint start = block % shape.z * TOPK_BLOCK;
    for (uint j = 0; j < TOPK_VALUES; ++j) {
        uint i = tid + j * TOPK_THREADS;
        uint token = start + i;
        local_values[j] = token < shape.x ? input[row * shape.x + token] : -INFINITY;
        local_ids[j] = token < shape.x ? token : INVALID_TOKEN;
    }
    for (uint span = 2; span <= TOPK_BLOCK; span *= 2) {
        for (uint stride = span / 2; stride > 0; stride /= 2) {
            if (stride >= SIMD_WIDTH) {
                for (uint j = 0; j < TOPK_VALUES; ++j) {
                    values[tid + j * TOPK_THREADS] = local_values[j];
                    ids[tid + j * TOPK_THREADS] = local_ids[j];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint j = 0; j < TOPK_VALUES; ++j) {
                uint i = tid + j * TOPK_THREADS;
                float other = stride < SIMD_WIDTH ? simd_shuffle_xor(local_values[j], ushort(stride)) : values[i ^ stride];
                uint other_id = stride < SIMD_WIDTH ? simd_shuffle_xor(local_ids[j], ushort(stride)) : ids[i ^ stride];
                bool first = ((i & span) == 0) == ((i & stride) == 0);
                if (candidate_before(other, other_id, local_values[j], local_ids[j]) == first) {
                    local_values[j] = other;
                    local_ids[j] = other_id;
                }
            }
            if (stride >= SIMD_WIDTH) threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < shape.y) {
        output[block * shape.y + tid] = local_values[0];
        output_ids[block * shape.y + tid] = local_ids[0];
    }
}

kernel void candidates_topk_merge(device const float* values [[buffer(0)]], device const uint* ids [[buffer(1)]],
                                 device float* output [[buffer(2)]], device uint* output_ids [[buffer(3)]],
                                 constant uint4& shape [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    uint k = shape.x;
    uint groups = shape.y;
    uint pairs = (groups + 1) / 2;
    uint row = index / (pairs * k * 2);
    uint pair = index / (k * 2) % pairs;
    uint local = index % (k * 2);
    bool left = local < k;
    uint source_group = pair * 2 + uint(!left);
    if (source_group >= groups) return;
    uint source = (row * groups + source_group) * k + local % k;
    uint other_group = pair * 2 + uint(left);
    uint low = 0;
    uint high = other_group < groups ? k : 0;
    float value = values[source];
    uint token = ids[source];
    while (low < high) {
        uint mid = low + (high - low) / 2;
        uint other = (row * groups + other_group) * k + mid;
        bool before = candidate_before(values[other], ids[other], value, token) || (ids[other] == token && !left);
        if (before) low = mid + 1;
        else high = mid;
    }
    uint rank = local % k + low;
    if (rank < k) {
        uint dest = (row * pairs + pair) * k + rank;
        output[dest] = value;
        output_ids[dest] = token;
    }
}
