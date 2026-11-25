#pragma once
// Minimal, self-contained MD5 (RFC 1321), header-only.
//
// Used only to derive a stable content hash for uploaded reference audio, so a
// re-upload of identical bytes maps to the same stored file name — matching the
// Python server's `hashlib.md5(content).hexdigest()` naming convention.
// Not used for any security purpose.

#include <cstdint>
#include <cstring>
#include <cctype>
#include <string>

namespace indextts {
namespace md5detail {

struct Ctx {
    uint32_t a, b, c, d;
    uint64_t len;       // total message length in bytes (before padding)
    uint8_t  buffer[64];
    uint32_t buflen;    // bytes currently buffered
};

inline uint32_t rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

inline void transform(Ctx& ctx, const uint8_t* block) {
    static const uint32_t K[64] = {
        0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
        0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
        0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
        0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
        0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
        0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
        0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
        0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
    };
    static const int S[64] = {
        7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
        5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
        4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
        6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
    };
    uint32_t M[16];
    for (int i = 0; i < 16; ++i) {
        M[i] =  (uint32_t)block[i*4]
             | ((uint32_t)block[i*4+1] << 8)
             | ((uint32_t)block[i*4+2] << 16)
             | ((uint32_t)block[i*4+3] << 24);
    }
    uint32_t A = ctx.a, B = ctx.b, C = ctx.c, D = ctx.d;
    for (int i = 0; i < 64; ++i) {
        uint32_t F; int g;
        if (i < 16)      { F = (B & C) | (~B & D);  g = i; }
        else if (i < 32) { F = (D & B) | (~D & C);  g = (5*i + 1) & 15; }
        else if (i < 48) { F = B ^ C ^ D;           g = (3*i + 5) & 15; }
        else             { F = C ^ (B | ~D);        g = (7*i) & 15; }
        uint32_t tmp = D;
        D = C;
        C = B;
        B = B + rotl(A + F + K[i] + M[g], S[i]);
        A = tmp;
    }
    ctx.a += A; ctx.b += B; ctx.c += C; ctx.d += D;
}

inline void init(Ctx& ctx) {
    ctx.a = 0x67452301; ctx.b = 0xefcdab89;
    ctx.c = 0x98badcfe; ctx.d = 0x10325476;
    ctx.len = 0; ctx.buflen = 0;
}

inline void update(Ctx& ctx, const uint8_t* data, size_t n) {
    ctx.len += n;
    while (n > 0) {
        uint32_t space = 64 - ctx.buflen;
        uint32_t take = (n < (size_t)space) ? (uint32_t)n : space;
        std::memcpy(ctx.buffer + ctx.buflen, data, take);
        ctx.buflen += take; data += take; n -= take;
        if (ctx.buflen == 64) { transform(ctx, ctx.buffer); ctx.buflen = 0; }
    }
}

inline void finalize(Ctx& ctx, uint8_t out[16]) {
    uint64_t bitlen = ctx.len * 8;   // capture true length before padding
    uint8_t pad = 0x80;
    update(ctx, &pad, 1);
    uint8_t zero = 0;
    while (ctx.buflen != 56) update(ctx, &zero, 1);
    uint8_t lenbytes[8];
    for (int i = 0; i < 8; ++i) lenbytes[i] = (uint8_t)((bitlen >> (8*i)) & 0xff);
    update(ctx, lenbytes, 8);
    uint32_t vals[4] = { ctx.a, ctx.b, ctx.c, ctx.d };
    for (int i = 0; i < 4; ++i) {
        out[i*4]   = (uint8_t)( vals[i]        & 0xff);
        out[i*4+1] = (uint8_t)((vals[i] >> 8)  & 0xff);
        out[i*4+2] = (uint8_t)((vals[i] >> 16) & 0xff);
        out[i*4+3] = (uint8_t)((vals[i] >> 24) & 0xff);
    }
}

} // namespace md5detail

inline std::string md5_hex(const void* data, size_t n) {
    md5detail::Ctx ctx;
    md5detail::init(ctx);
    md5detail::update(ctx, static_cast<const uint8_t*>(data), n);
    uint8_t digest[16];
    md5detail::finalize(ctx, digest);
    static const char* hexdig = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[i*2]   = hexdig[(digest[i] >> 4) & 0xF];
        out[i*2+1] = hexdig[digest[i] & 0xF];
    }
    return out;
}

inline std::string md5_hex(const std::string& s) {
    return md5_hex(s.data(), s.size());
}

} // namespace indextts
