/* C port of https://github.com/microsoft/qlib/blob/main/qlib/data/_libs/rolling.pyx
 * Copyright (c) Microsoft Corporation. Licensed under MIT; see LICENSE.
 */
#include "rolling.h"

#include <math.h>

enum rolling_kind { MEAN, SLOPE, RSQUARE, RESI };

static int rolling(const double *input, size_t length, size_t window,
                   double *output, enum rolling_kind kind)
{
    size_t i, count = 0;
    double x_sum = 0.0, x2_sum = 0.0;
    double y_sum = 0.0, y2_sum = 0.0, xy_sum = 0.0;
    double w;

    if (window == 0 || (length != 0 && (input == NULL || output == NULL))) {
        return 1;
    }
    if (length == 0) {
        return 0;
    }
    /* Translating the time axis leaves all four statistics unchanged. */
    if (window > length) {
        window = length;
    }
    w = (double)window;

    for (i = 0; i < length; ++i) {
        const double value = input[i];
        if (kind != MEAN) {
            /* Shift existing observations left by one before eviction.
             * The departing observation is now at x=0.
             */
            xy_sum -= y_sum;
            x2_sum += (double)count - 2.0 * x_sum;
            x_sum -= (double)count;
        }
        if (i >= window && !isnan(input[i - window])) {
            const double old = input[i - window];
            --count;
            y_sum -= old;
            if (kind == RSQUARE) {
                y2_sum -= old * old;
            }
        }
        /* Clear accumulated roundoff when the window becomes empty. */
        if (count == 0) {
            x_sum = x2_sum = y_sum = y2_sum = xy_sum = 0.0;
        }
        if (!isnan(value)) {
            ++count;
            y_sum += value;
            if (kind != MEAN) {
                x_sum += w;
                x2_sum += w * w;
                xy_sum += w * value;
            }
            if (kind == RSQUARE) {
                y2_sum += value * value;
            }
        }

        output[i] = NAN;
        if (kind == MEAN) {
            if (count != 0) {
                output[i] = y_sum / (double)count;
            }
        } else if (count >= 2) {
            const double n = (double)count;
            const double xx = n * x2_sum - x_sum * x_sum;
            const double xy = n * xy_sum - x_sum * y_sum;
            if (xx > 0.0) {
                const double slope = xy / xx;
                if (kind == SLOPE) {
                    output[i] = slope;
                } else if (kind == RESI) {
                    const double intercept = y_sum / n - slope * x_sum / n;
                    output[i] = value - (slope * w + intercept);
                } else {
                    const double yy = n * y2_sum - y_sum * y_sum;
                    if (yy > 0.0) {
                        const double r = xy / sqrt(xx * yy);
                        output[i] = r * r;
                    }
                }
            }
        }
    }
    return 0;
}

int qlib_rolling_mean(const double *input, size_t length, size_t window, double *output)
{
    return rolling(input, length, window, output, MEAN);
}

int qlib_rolling_slope(const double *input, size_t length, size_t window, double *output)
{
    return rolling(input, length, window, output, SLOPE);
}

int qlib_rolling_rsquare(const double *input, size_t length, size_t window, double *output)
{
    return rolling(input, length, window, output, RSQUARE);
}

int qlib_rolling_resi(const double *input, size_t length, size_t window, double *output)
{
    return rolling(input, length, window, output, RESI);
}
