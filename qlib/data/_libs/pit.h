#ifndef QLIB_PIT_H
#define QLIB_PIT_H
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define QLIB_PIT_API __declspec(dllexport)
#else
#define QLIB_PIT_API
#endif
/* Dates are sorted within each [start, start+count) group.
 * Output is the last record position with date <= asof, or -1.
 * Uses aligned arrays, independent of the packed on-disk record layout.
 */
QLIB_PIT_API int qlib_pit_asof(const uint32_t *dates, size_t length,
    const uint64_t *starts, const uint64_t *counts, size_t groups,
    uint32_t asof, int64_t *output);
#endif
