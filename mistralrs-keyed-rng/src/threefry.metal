#include <metal_stdlib>
using namespace metal;
inline uint rotl32(uint x, uint r) {
    return (x << r) | (x >> (32u - r));
}
inline uint2 threefry_round(uint x0_in, uint x1_in, uint rotation) {
    const auto x0 = x0_in + x1_in;
    const auto x1 = rotl32(x1_in, rotation) ^ x0;

    return uint2(x0, x1);
}
inline uint2 threefry2x32(uint counter0, uint counter1, uint key0, uint key1) {
    const auto parity = 0x1bd11bdau;

    const auto ks0 = key0;
    const auto ks1 = key1;
    const auto ks2 = parity ^ ks0 ^ ks1;

    auto x0 = counter0 + ks0;
    auto x1 = counter1 + ks1;

    auto pair = threefry_round(x0, x1, 13u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 15u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 26u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 6u);
    x0 = pair.x;
    x1 = pair.y;

    x0 = x0 + ks1;
    x1 = x1 + ks2 + 1u;

    pair = threefry_round(x0, x1, 17u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 29u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 16u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 24u);
    x0 = pair.x;
    x1 = pair.y;

    x0 = x0 + ks2;
    x1 = x1 + ks0 + 2u;

    pair = threefry_round(x0, x1, 13u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 15u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 26u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 6u);
    x0 = pair.x;
    x1 = pair.y;

    x0 = x0 + ks0;
    x1 = x1 + ks1 + 3u;

    pair = threefry_round(x0, x1, 17u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 29u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 16u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 24u);
    x0 = pair.x;
    x1 = pair.y;

    x0 = x0 + ks1;
    x1 = x1 + ks2 + 4u;

    pair = threefry_round(x0, x1, 13u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 15u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 26u);
    x0 = pair.x;
    x1 = pair.y;

    pair = threefry_round(x0, x1, 6u);
    x0 = pair.x;
    x1 = pair.y;

    x0 = x0 + ks2;
    x1 = x1 + ks0 + 5u;

    return uint2(x0, x1);
}
inline float uniform_from_uint(uint value) {
    return float(value >> 8u) * 5.960464477539063e-8;
}
