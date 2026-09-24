#include "pit.h"

int qlib_pit_asof(const uint32_t *dates, size_t length,
    const uint64_t *starts, const uint64_t *counts, size_t groups,
    uint32_t asof, int64_t *output)
{
    size_t i;
    if ((length && dates == NULL) || length > INT64_MAX ||
        (groups && (starts == NULL || counts == NULL || output == NULL))) {
        return 1;
    }
    for (i = 0; i < groups; ++i) {
        uint64_t lo = starts[i], hi;
        if (lo > length || counts[i] > length - lo) {
            return 1;
        }
        hi = lo + counts[i];
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (dates[mid] <= asof) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        output[i] = lo == starts[i] ? -1 : (int64_t)(lo - 1);
    }
    return 0;
}
