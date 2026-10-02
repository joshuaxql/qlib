/* Based on https://github.com/microsoft/qlib/blob/main/qlib/data/_libs/expanding.pyx
 * Copyright (c) Microsoft Corporation. Licensed under MIT; see LICENSE.
 */
#include "expanding.h"
#include "stable_stats.h"
static int expanding(const double *input, const double *left, size_t length,
                     double *output, enum statistic_kind kind)
{
    size_t i;
    moments aggregate = {0};
    if (length != 0 && (input == NULL || output == NULL)) { return 1; }
    for (i = 0; i < length; ++i) {
        const double x = left == NULL ? (double)i : left[i];
        aggregate = combine(aggregate, observation(x, input[i], left == NULL));
        output[i] = statistic(aggregate, kind, x, input[i]);
    }
    return 0;
}
#define UNIVARIATE(name, kind) \
    int qlib_expanding_##name(const double *input, size_t length, double *output) \
    { return expanding(input, NULL, length, output, kind); }
UNIVARIATE(mean, MEAN)
UNIVARIATE(slope, SLOPE)
UNIVARIATE(rsquare, RSQUARE)
UNIVARIATE(resi, RESI)
#define BIVARIATE(name, kind) \
    int qlib_expanding_##name(const double *left, const double *right, size_t length, double *output) \
    { \
        if (length != 0 && left == NULL) { return 1; } \
        return expanding(right, left, length, output, kind); \
    }
BIVARIATE(corr, CORR)
BIVARIATE(cov, COV)
