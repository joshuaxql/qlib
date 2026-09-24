/* C implementation of microsoft/qlib's expanding.pyx interface. See LICENSE. */
#ifndef QLIB_EXPANDING_H
#define QLIB_EXPANDING_H

#include <stddef.h>

#if defined(_WIN32)
#define QLIB_EXPANDING_API __declspec(dllexport)
#else
#define QLIB_EXPANDING_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Input/output are non-overlapping arrays of length doubles.
 * Each output uses all observations from index 0 through the current index.
 * NaNs are ignored, but keep their position on the regression time axis.
 * Returns 0 on success, 1 on invalid arguments. Empty arrays may be NULL.
 * No allocation, global state, or Python runtime is required.
 */
QLIB_EXPANDING_API int qlib_expanding_mean(const double *input, size_t length, double *output);
QLIB_EXPANDING_API int qlib_expanding_slope(const double *input, size_t length, double *output);
QLIB_EXPANDING_API int qlib_expanding_rsquare(const double *input, size_t length, double *output);
QLIB_EXPANDING_API int qlib_expanding_resi(const double *input, size_t length, double *output);

#ifdef __cplusplus
}
#endif
#endif
