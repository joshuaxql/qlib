/* C port of microsoft/qlib's rolling.pyx. See LICENSE. */
#ifndef QLIB_ROLLING_H
#define QLIB_ROLLING_H

#include <stddef.h>

#if defined(_WIN32)
#define QLIB_API __declspec(dllexport)
#else
#define QLIB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Input/output are non-overlapping arrays of length doubles. window > 0.
 * NaNs are ignored, but keep their position in regression windows.
 * Partial windows are evaluated from the first observation (min_periods=1).
 * Returns 0 on success, 1 on invalid arguments. Empty arrays may be NULL.
 * No allocation, global state, or Python runtime is required.
 */
QLIB_API int qlib_rolling_mean(const double *input, size_t length,
                              size_t window, double *output);
QLIB_API int qlib_rolling_slope(const double *input, size_t length,
                               size_t window, double *output);
QLIB_API int qlib_rolling_rsquare(const double *input, size_t length,
                                 size_t window, double *output);
QLIB_API int qlib_rolling_resi(const double *input, size_t length,
                              size_t window, double *output);

#ifdef __cplusplus
}
#endif
#endif
