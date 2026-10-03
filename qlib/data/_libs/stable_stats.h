/* Centered, binary-scaled moments shared by the rolling/expanding kernels. */
#ifndef QLIB_STABLE_STATS_H
#define QLIB_STABLE_STATS_H
#include <math.h>
#include <stddef.h>

enum statistic_kind { MEAN, SLOPE, RSQUARE, RESI, CORR, COV };
typedef struct {
    size_t count;
    int x_exponent, y_exponent;
    unsigned int infinity;
    double x_anchor, y_anchor, x_offset, y_offset, xx, yy, xy;
} moments;

static double rescale_moment(double value, int shift)
{
    /* Equal exponents need no libm call. Preserve the original scalbn path
     * for every nonzero shift, including subnormal/overflow handling.
     */
    return shift == 0 ? value : scalbn(value, shift);
}

static moments observation(double x, double y, int time_axis)
{
    moments result = {0};
    int exponent;
    if (isnan(x) || isnan(y)) { return result; }
    if (isinf(x) || isinf(y)) {
        result.infinity = (isinf(x) ? 4u : 0u) |
                          (isinf(y) ? (y > 0.0 ? 1u : 2u) : 0u);
        return result;
    }
    result.count = 1;
    /* Binary scaling preserves representable differences at a large baseline.
     * Zero must not force a tiny nonzero series back to unit scale.
     */
    if (time_axis) { result.x_exponent = 0; }
    else if (x == 0.0) { result.x_exponent = -1075; }
    else { (void)frexp(x, &exponent); result.x_exponent = exponent - 1; }
    if (y == 0.0) { result.y_exponent = -1075; }
    else { (void)frexp(y, &exponent); result.y_exponent = exponent - 1; }
    result.x_anchor = scalbn(x, -result.x_exponent);
    result.y_anchor = scalbn(y, -result.y_exponent);
    return result;
}

static moments combine(moments a, moments b)
{
    moments result = {0};
    double ax, ay, bx, by, amx, amy, bmx, bmy, dx, dy, fraction, weight;
    if (a.count == 0) { b.infinity |= a.infinity; return b; }
    if (b.count == 0) { a.infinity |= b.infinity; return a; }
    result.count = a.count + b.count;
    result.infinity = a.infinity | b.infinity;
    result.x_exponent = a.x_exponent > b.x_exponent ? a.x_exponent : b.x_exponent;
    result.y_exponent = a.y_exponent > b.y_exponent ? a.y_exponent : b.y_exponent;
    ax = rescale_moment(a.x_anchor, a.x_exponent - result.x_exponent);
    ay = rescale_moment(a.y_anchor, a.y_exponent - result.y_exponent);
    bx = rescale_moment(b.x_anchor, b.x_exponent - result.x_exponent);
    by = rescale_moment(b.y_anchor, b.y_exponent - result.y_exponent);
    amx = rescale_moment(a.x_offset, a.x_exponent - result.x_exponent);
    amy = rescale_moment(a.y_offset, a.y_exponent - result.y_exponent);
    bmx = rescale_moment(b.x_offset, b.x_exponent - result.x_exponent);
    bmy = rescale_moment(b.y_offset, b.y_exponent - result.y_exponent);
    /* Keep the mean relative to an observed anchor. Reconstructing a large
     * absolute mean here would round away fractional mean increments.
     */
    dx = (bx - ax) + bmx - amx; dy = (by - ay) + bmy - amy;
    fraction = (double)b.count / (double)result.count;
    weight = (double)a.count * fraction;
    result.x_anchor = ax; result.y_anchor = ay;
    result.x_offset = amx + dx * fraction;
    result.y_offset = amy + dy * fraction;
    result.xx = rescale_moment(a.xx, 2 * (a.x_exponent - result.x_exponent)) +
                rescale_moment(b.xx, 2 * (b.x_exponent - result.x_exponent)) + dx * dx * weight;
    result.yy = rescale_moment(a.yy, 2 * (a.y_exponent - result.y_exponent)) +
                rescale_moment(b.yy, 2 * (b.y_exponent - result.y_exponent)) + dy * dy * weight;
    result.xy = rescale_moment(a.xy, a.x_exponent + a.y_exponent -
                            result.x_exponent - result.y_exponent) +
                rescale_moment(b.xy, b.x_exponent + b.y_exponent -
                            result.x_exponent - result.y_exponent) + dx * dy * weight;
    return result;
}

static double statistic(moments value, enum statistic_kind kind, double x, double y)
{
    double correlation, residual;
    if (kind == MEAN) {
        if ((value.infinity & 3u) == 3u) { return NAN; }
        if (value.infinity & 1u) { return INFINITY; }
        if (value.infinity & 2u) { return -INFINITY; }
        return value.count ? scalbn(value.y_anchor + value.y_offset, value.y_exponent) : NAN;
    }
    if (value.infinity || value.count < 2) { return NAN; }
    if (kind == COV) {
        return scalbn(value.xy / (double)(value.count - 1),
                      value.x_exponent + value.y_exponent);
    }
    if (value.xx <= 0.0) { return NAN; }
    if (kind == SLOPE) {
        return scalbn(value.xy / value.xx, value.y_exponent - value.x_exponent);
    }
    if (kind == RESI) {
        if (!isfinite(y)) { return NAN; }
        residual = (scalbn(y, -value.y_exponent) - value.y_anchor - value.y_offset) -
                   value.xy / value.xx *
                   (scalbn(x, -value.x_exponent) - value.x_anchor - value.x_offset);
        return scalbn(residual, value.y_exponent);
    }
    if (value.yy <= 0.0) { return NAN; }
    correlation = (value.xy / sqrt(value.xx)) / sqrt(value.yy);
    return kind == CORR ? correlation : correlation * correlation;
}
#endif
