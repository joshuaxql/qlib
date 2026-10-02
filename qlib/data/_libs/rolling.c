/* C port of https://github.com/microsoft/qlib/blob/main/qlib/data/_libs/rolling.pyx
 * Copyright (c) Microsoft Corporation. Licensed under MIT; see LICENSE.
 */
#include "rolling.h"
#include "stable_stats.h"
#include <stdint.h>
#include <stdlib.h>

static int rolling(const double *input, const double *left, size_t length,
                   size_t window, double *output, enum statistic_kind kind)
{
    size_t i, front_count = 0, back_count = 0;
    moments *front, *back, *leaves;
    if (window == 0 || (length != 0 && (input == NULL || output == NULL))) { return 1; }
    if (length == 0) { return 0; }
    if (window > length) { window = length; }
    if (window > SIZE_MAX / sizeof(moments) / 3) { return 2; }
    front = malloc(3 * window * sizeof(moments));
    if (front == NULL) { return 2; }
    back = front + window;
    leaves = back + window;
    for (i = 0; i < length; ++i) {
        moments aggregate = {0};
        const double x = left == NULL ? (double)i : left[i];
        const moments leaf = observation(x, input[i], left == NULL);
        /* Aggregate stacks form a queue: O(length) amortized, O(window) space.
         * Eviction never subtracts moments contaminated by an old outlier.
         */
        if (i >= window) {
            if (front_count == 0) {
                while (back_count != 0) {
                    const moments item = leaves[--back_count];
                    front[front_count] = front_count == 0 ? item :
                                         combine(item, front[front_count - 1]);
                    ++front_count;
                }
            }
            --front_count;
        }
        leaves[back_count] = leaf;
        back[back_count] = back_count == 0 ? leaf : combine(back[back_count - 1], leaf);
        ++back_count;
        if (front_count != 0) { aggregate = front[front_count - 1]; }
        aggregate = combine(aggregate, back[back_count - 1]);
        output[i] = statistic(aggregate, kind, x, input[i]);
    }
    free(front);
    return 0;
}
#define UNIVARIATE(name, kind) \
    int qlib_rolling_##name(const double *input, size_t length, size_t window, double *output) \
    { return rolling(input, NULL, length, window, output, kind); }
UNIVARIATE(mean, MEAN)
UNIVARIATE(slope, SLOPE)
UNIVARIATE(rsquare, RSQUARE)
UNIVARIATE(resi, RESI)
#define BIVARIATE(name, kind) \
    int qlib_rolling_##name(const double *left, const double *right, size_t length, \
                            size_t window, double *output) \
    { \
        if (length != 0 && left == NULL) { return 1; } \
        return rolling(right, left, length, window, output, kind); \
    }
BIVARIATE(corr, CORR)
BIVARIATE(cov, COV)
