#include <stddef.h>
#include <stdint.h>

void split_fields(const uint8_t *restrict raw, uint8_t *restrict fields, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const uint8_t low = raw[2 * i];
        const uint8_t high = raw[2 * i + 1];
        fields[i] = ((high & 127) << 1) | (low >> 7);
        fields[count + i] = (high & 128) | (low & 127);
    }
}

static inline void write_word(uint8_t *output, uint8_t exponent, uint8_t sign_fraction) {
    output[0] = (sign_fraction & 127) | ((exponent & 1) << 7);
    output[1] = (sign_fraction & 128) | (exponent >> 1);
}

void join_fields(const uint8_t *restrict exponent, const uint8_t *restrict sign_fraction,
                 uint8_t *restrict raw, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        write_word(&raw[2 * i], exponent[i], sign_fraction[i]);
    }
}

void join_delta(const uint8_t *restrict base_exponent, const uint8_t *restrict exponent_xor,
                const uint8_t *restrict sign_fraction, uint8_t *restrict raw, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        write_word(&raw[2 * i], base_exponent[i] ^ exponent_xor[i], sign_fraction[i]);
    }
}
