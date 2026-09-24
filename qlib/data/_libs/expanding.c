/* Based on https://github.com/microsoft/qlib/blob/main/qlib/data/_libs/expanding.pyx
 * Copyright (c) Microsoft Corporation. Licensed under MIT; see LICENSE.
 */
#include "expanding.h"

#include <math.h>

enum expanding_kind { MEAN, SLOPE, RSQUARE, RESI };

static int expanding(const double *input, size_t length, double *output,
                     enum expanding_kind kind)
{
    size_t i, count = 0;
    double y_sum = 0.0, x_mean = 0.0, y_mean = 0.0;
    double xx = 0.0, yy = 0.0, xy = 0.0;

    if (length != 0 && (input == NULL || output == NULL)) {
        return 1;
    }
    for (i = 0; i < length; ++i) {
        const double value = input[i];
        const double x = (double)i + 1.0;
        if (!isnan(value)) {
            ++count;
            if (kind == MEAN) {
                y_sum += value;
            } else if (count == 1) {
                x_mean = x;
                y_mean = value;
            } else {
                /* Online centered moments avoid subtracting large raw sums
                 * when computing regression variance and covariance.
                 */
                const double dx = x - x_mean;
                const double dy = value - y_mean;
                x_mean += dx / (double)count;
                y_mean += dy / (double)count;
                xx += dx * (x - x_mean);
                xy += dx * (value - y_mean);
                if (kind == RSQUARE) {
                    yy += dy * (value - y_mean);
                }
            }
        }

        output[i] = NAN;
        if (kind == MEAN) {
            if (count != 0) {
                output[i] = y_sum / (double)count;
            }
        } else if (count >= 2 && xx > 0.0) {
            const double slope = xy / xx;
            if (kind == SLOPE) {
                output[i] = slope;
            } else if (kind == RESI) {
                output[i] = value - (y_mean + slope * (x - x_mean));
            } else if (yy > 0.0) {
                const double r = (xy / sqrt(xx)) / sqrt(yy);
                output[i] = r * r;
            }
        }
    }
    return 0;
}

int qlib_expanding_mean(const double *input, size_t length, double *output)
{
    return expanding(input, length, output, MEAN);
}

int qlib_expanding_slope(const double *input, size_t length, double *output)
{
    return expanding(input, length, output, SLOPE);
}

int qlib_expanding_rsquare(const double *input, size_t length, double *output)
{
    return expanding(input, length, output, RSQUARE);
}

int qlib_expanding_resi(const double *input, size_t length, double *output)
{
    return expanding(input, length, output, RESI);
}
