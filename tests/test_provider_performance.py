"""Equivalence and bounded-work checks for provider/query optimizations."""
import ast
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.data import LocalProvider
from qlib.data.base import ExpressionEngine, _expression_tree, history_bounds
from qlib.data.pit import HEADER, RECORD_DTYPE, StockPIT, register_fields, write_stock

A, B = "000001.SZ", "600000.SH"


class ProviderPerformanceTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dates = pd.bdate_range("2024-01-02", periods=80, name="datetime")
        for directory in ("calendars", "instruments", f"features/{A}", f"features/{B}"):
            (self.root / directory).mkdir(parents=True)
        (self.root / "calendars/day.txt").write_text("\n".join(self.dates.strftime("%Y-%m-%d")))
        (self.root / "instruments/all.txt").write_text("".join(
            f"{code} {self.dates[0].date()} {self.dates[-1].date()}\n" for code in (A, B)))
        pd.DataFrame({"ts_code": [A, B], "list_date": ["2000-01-01"] * 2}).to_csv(
            self.root / "stock_basic.csv", index=False)
        self.raw = {}
        positions = np.arange(len(self.dates))
        for number, code in enumerate((A, B)):
            close = 10 + number * 8 + positions / 3 + np.sin(positions)
            close[[8, 21]] = np.nan
            volume = 1000 + number * 200 + positions ** 2
            volume = volume.astype(float)
            volume[33] = np.inf
            factor = np.repeat([1.0, 2.0, 4.0, 8.0], 20)
            for field, values in (("close", close), ("volume", volume), ("factor", factor)):
                values = np.asarray(values, dtype="<f4")
                self.raw[code, field] = values.astype(float)
                np.r_[np.float32(0), values].astype("<f4").tofile(
                    self.root / "features" / code / f"{field}.day.bin")
        self.provider = LocalProvider(self.root)
        self.financial = self.root / "financial"
        self.ids = register_fields(self.financial, {name: {"source": "fina_indicator_vip"}
                                                   for name in ("eps", "roe", "other.asset")})
        self.rows = pd.DataFrame([
            (20240103, 202301, "eps", 1.0), (20240103, 202303, "eps", 3.0),
            (20240110, 202304, "eps", 4.0), (20240115, 202303, "eps", np.nan),
            (20240116, 202304, "eps", np.nan), (20240117, 202304, "eps", 5.0),
            (20240104, 202301, "roe", 10.0), (20240118, 202304, "roe", 40.0),
            (20240105, 202101, "other.asset", 900.0),
        ], columns=["date", "period", "field", "value"])
        write_stock(self.financial, A, self.rows, update=False)

    def full_history(self, provider, fields, start, end, **kwargs):
        with patch("qlib.data.base.history_bounds", return_value=None):
            return provider.features([A], fields, start, end, **kwargs)

    def assert_same(self, actual, expected):
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=2e-12, atol=2e-12)

    def test_universe_spans_match_independent_datewise_membership(self):
        spans = {
            B: [(self.dates[8], self.dates[15]), (self.dates[12], self.dates[20]),
                (self.dates[40], pd.NaT), (self.dates[70], self.dates[60])],
            A: [("1990-01-01", self.dates[3]), (self.dates[25], "2099-01-01")],
            "000002.SZ": [],
        }
        actual = self.provider.universe(spans, self.dates[2], self.dates[50])
        expected = pd.DataFrame({code: [any(day >= pd.Timestamp(start) and day <= pd.Timestamp(end)
                                             for start, end in intervals) for day in actual.index]
                                 for code, intervals in sorted(spans.items())}, index=actual.index)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertTrue(self.provider.universe(spans, "2030-01-01", "2030-01-02").empty)

    def test_instrument_list_preserves_filtered_membership_without_building_spans(self):
        instruments = self.provider.instruments("all", [{"filter_type": "ExpressionFilter",
                                                       "expression": "$close > 30"}])
        for pool in (instruments, {A: [(self.dates[2], self.dates[5])], B: []}):
            expected = list(self.provider.list_instruments(pool, self.dates[0], self.dates[40]))
            with patch("qlib.data.data.np.diff", side_effect=AssertionError("Unneeded span reconstruction")):
                actual = self.provider.list_instruments(pool, self.dates[0], self.dates[40], as_list=True)
            self.assertEqual(actual, expected)
        self.assertEqual(self.provider.list_instruments("all", "2030-01-01", "2030-01-02", as_list=True), [])

    def test_cached_syntax_keys_do_not_cache_stock_values_or_policy(self):
        expression = "Power($close + $volume / 7, 2) - Ref($close, 3)"
        _expression_tree.cache_clear()
        with patch("qlib.data.base.ast.dump", wraps=ast.dump) as dump:
            first = ExpressionEngine(self.provider, A, self.dates, False).evaluate(expression)
            initial_calls = dump.call_count
            second = ExpressionEngine(self.provider, B, self.dates, False).evaluate(expression)
            self.assertGreater(initial_calls, 0)
            self.assertEqual(dump.call_count, initial_calls)
        self.assertFalse(first.equals(second))
        ExpressionEngine(self.provider, A, self.dates, True).evaluate("Ref($close, -1)")
        with self.assertRaises(ValueError):
            ExpressionEngine(self.provider, A, self.dates, False).evaluate("Ref($close, -1)")
        changed = self.raw[A, "close"].copy() + 5
        np.r_[0, changed].astype("<f4").tofile(self.root / "features" / A / "close.day.bin")
        self.provider.clear_cache()
        new = ExpressionEngine(self.provider, A, self.dates, False).evaluate(expression)
        self.assertFalse(first.equals(new))

    def test_nested_fixed_windows_evaluate_only_the_required_history(self):
        fields = ["Min(Ref($close, 3), 5)", "Max(Delta($close, 2), 7)",
                  "IdxMax(Min($close, 3), 5)", "Count(Ref($close, 2) > 20, 6)",
                  "IdxMin($close, 5)", "Min(If($close > 20, $close, 0), 4)"]
        self.assertEqual(history_bounds(fields, False), (8, 0))
        lengths = []

        class ObservedEngine(ExpressionEngine):
            def __init__(engine, provider, instrument, calendar, allow_future=True):
                lengths.append(len(calendar))
                super().__init__(provider, instrument, calendar, allow_future)

        with patch("qlib.data.base.ExpressionEngine", ObservedEngine):
            actual = self.provider.features([A], fields, self.dates[45], self.dates[49], allow_future=False)
        self.assertEqual(lengths, [13])
        expected = self.full_history(self.provider, fields, self.dates[45], self.dates[49], allow_future=False)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_future_dependencies_keep_sufficient_right_history(self):
        fields = ["Min(Ref($close, -3), 5)", "Ref(Max($close, 7), -2)",
                  "Delta(Ref($close, -2), 4)", "IdxMin(Ref($close, 2), 5)"]
        self.assertEqual(history_bounds(fields), (6, 3))
        for start, end in ((45, 49), (77, 79), (0, 2)):
            actual = self.provider.features([A], fields, self.dates[start], self.dates[end], adjust="qfq")
            expected = self.full_history(self.provider, fields, self.dates[start], self.dates[end], adjust="qfq")
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_statistical_windows_preserve_full_history_and_float_bits(self):
        # Even sub-ulp changes can reorder near-zero residuals after regression.
        # A rolling window's mathematical horizon does not prove bit identity.
        expressions = [f"{name}($close, 7)" for name in
                       ("Mean", "Sum", "Std", "Var", "Med", "Skew", "Kurt", "WMA",
                        "Rank", "Slope", "Rsquare", "Resi")]
        expressions += ["Quantile($close, 7, 0.5)", "Corr($close, $volume, 7)",
                        "Cov($close, $volume, 7)", "Mean(Ref($close, 3), 5)",
                        "Ref(Std($close, 7), -2)"]
        lengths = []

        class ObservedEngine(ExpressionEngine):
            def __init__(engine, provider, instrument, calendar, allow_future=True):
                lengths.append(len(calendar))
                super().__init__(provider, instrument, calendar, allow_future)

        for expression in expressions:
            with self.subTest(expression=expression):
                self.assertIsNone(history_bounds([expression]))
                lengths.clear()
                with patch("qlib.data.base.ExpressionEngine", ObservedEngine):
                    actual = self.provider.features([A], [expression], self.dates[45], self.dates[49])
                self.assertEqual(lengths, [len(self.dates)])
                expected = self.full_history(self.provider, [expression], self.dates[45], self.dates[49])
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_expanding_ema_first_observation_and_dynamic_windows_keep_origin(self):
        for expression in ("Mean($close, 0)", "Sum($close, 0)", "EMA($close, 5)",
                           "Mean(EMA($close, 5), 3)", "Ref($close, 0)", "Delta($close, 0)",
                           "Mean($close, 2 + 1)"):
            with self.subTest(expression=expression):
                self.assertIsNone(history_bounds([expression]))
                actual = self.provider.features([A], [expression], self.dates[45], self.dates[49])
                expected = self.full_history(self.provider, [expression], self.dates[45], self.dates[49])
                self.assert_same(actual, expected)
        self.assertIsNone(history_bounds(["EMA($close, 5)", "Slope($close, 3)"]))

    def test_signal_qfq_anchor_segments_equal_independent_daily_anchors(self):
        provider = self.provider._price_view("qfq")._signal_view()
        fields = ["Mean($close + 1, 5)", "Slope($close, 4)", "$close > 30", "P($$eps) / $close"]
        actual = provider.features([A], fields, self.dates[17], self.dates[43], allow_future=False)
        parts = [self.full_history(self.provider, fields, day, day, allow_future=False, adjust="qfq")
                 for day in self.dates[17:44]]
        self.assert_same(actual, pd.concat(parts))
        factor = self.raw[A, "factor"].copy()
        factor[60:] *= 8
        np.r_[0, factor].astype("<f4").tofile(self.root / "features" / A / "factor.day.bin")
        self.provider.clear_cache()
        self.assert_same(actual, provider.features([A], fields, self.dates[17], self.dates[43], allow_future=False))

    def test_direct_daily_fields_match_independent_price_and_finite_oracle(self):
        fields = ["close", "open", "high", "volume", "factor"]
        close = self.raw[A, "close"].copy()
        close[5], close[36] = np.inf, -np.inf
        factor = self.raw[A, "factor"].copy()
        factor[:6] = [np.nan, 0, -1, np.inf, -np.inf, 0]
        factor[[18, 19, 35, 37]] = [0, -2, np.nan, np.inf]
        for name, values in (("close", close), ("open", close + 1), ("high", close + 2), ("factor", factor)):
            np.r_[0, values].astype("<f4").tofile(self.root / "features" / A / f"{name}.day.bin")
            self.raw[A, name] = values.astype("<f4").astype(float)
        self.provider.clear_cache()
        for adjust in ("hfq", "qfq", "none"):
            for end in (self.dates[4], self.dates[19], self.dates[40], None):
                with self.subTest(adjust=adjust, end=end):
                    dates = self.provider.calendar(self.dates[2], end)
                    valid = np.isfinite(factor) & (factor > 0)
                    multiplier = np.where(valid, factor, np.nan)
                    if adjust == "qfq":
                        stop = len(factor) if end is None else self.dates.searchsorted(end, side="right")
                        visible = factor[:stop][valid[:stop]]
                        multiplier = multiplier / (visible[-1] if len(visible) else np.nan)
                    expected = {}
                    for name in fields:
                        values = self.raw[A, name].copy()
                        if name in ("close", "open", "high") and adjust != "none":
                            values *= multiplier
                        values[~np.isfinite(values)] = np.nan
                        expected[name] = pd.Series(values, index=self.dates).reindex(dates)
                    expected = pd.concat({A: pd.DataFrame(expected)}, names=["instrument", "datetime"])
                    actual = self.provider.daily([A], fields, self.dates[2], end, adjust=adjust)
                    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
                    with patch.object(self.provider, "_direct_daily_frame", return_value=None):
                        legacy = self.provider.daily([A], fields, self.dates[2], end, adjust=adjust)
                    pd.testing.assert_frame_equal(actual, legacy, check_exact=True)
        with patch.object(self.provider, "_daily", wraps=self.provider._daily) as reads:
            self.provider.daily([A], fields, self.dates[40], self.dates[45])
        self.assertEqual([call.args[1] for call in reads.call_args_list], fields)

    def test_direct_daily_fields_preserve_hooks_virtual_fields_and_causal_qfq(self):
        class ChangedFields(LocalProvider):
            def _field(provider, code, field):
                return super()._field(code, field) + (100 if field == "close" else 0)

        class ChangedDaily(LocalProvider):
            def _read_daily(provider, code, field):
                return super()._read_daily(code, field) + (200 if field == "volume" else 0)

        for provider in (ChangedFields(self.root), ChangedDaily(self.root)):
            actual = provider.daily([A], ["close", "volume"], self.dates[30], self.dates[35])
            with patch.object(provider, "_direct_daily_frame", return_value=None):
                expected = provider.daily([A], ["close", "volume"], self.dates[30], self.dates[35])
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        (self.root / "instruments/st.txt").write_text(
            f"{A} {self.dates[10].date()} {self.dates[20].date()}\n")
        self.provider.clear_cache()
        fields = ["$close", "$is_st", "$list_days"]
        actual = self.provider.features([A], fields, self.dates[12], self.dates[23])
        with patch.object(self.provider, "_direct_daily_frame", return_value=None):
            expected = self.provider.features([A], fields, self.dates[12], self.dates[23])
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        causal = self.provider._price_view("qfq")._signal_view()
        fields = ["$close", "$volume", "$factor"]
        actual = causal.features([A], fields, self.dates[17], self.dates[43], allow_future=False)
        with patch.object(causal, "_direct_daily_frame", return_value=None):
            expected = pd.concat([self.full_history(causal, fields, day, day, allow_future=False, adjust="qfq")
                                  for day in self.dates[17:44]])
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    def test_pit_selected_blocks_snapshots_and_events_match_source_oracle(self):
        stock = self.provider._pit.stock(A)
        selected_ids = [self.ids["roe"], self.ids["eps"], self.ids["eps"], 99999]
        expected_rows = self.rows[self.rows.field.isin(["eps", "roe"])].assign(
            field_id=lambda frame: frame.field.map(self.ids)).sort_values(["field_id", "period", "date"])
        selected = stock.select(selected_ids)
        for name in ("field_id", "date", "period", "value"):
            np.testing.assert_allclose(selected[name], expected_rows[name], equal_nan=True)
        self.assertIsNone(stock._event_order)
        before_size = stock.nbytes
        for date in self.dates[:16]:
            cutoff = date.year * 10000 + date.month * 100 + date.day
            visible = expected_rows[expected_rows.date <= cutoff]
            snapshot = visible.sort_values("date").drop_duplicates(["field_id", "period"], keep="last").sort_values(
                ["field_id", "period"])
            actual = stock.snapshot(selected_ids, date)
            for name in ("field_id", "date", "period", "value"):
                np.testing.assert_allclose(actual[name], snapshot[name], equal_nan=True)
        events = list(stock.events(selected_ids, self.dates[15]))
        for (actual_date, actual), (expected_date, expected) in zip(events, expected_rows.groupby("date", sort=True)):
            self.assertEqual(actual_date, expected_date)
            for name in ("field_id", "date", "period", "value"):
                np.testing.assert_allclose(actual[name], expected[name], equal_nan=True)
        self.assertEqual(len(events), expected_rows.date.nunique())
        self.assertIsNone(stock._event_order)
        self.assertEqual(stock.nbytes, before_size)
        self.assertFalse(stock.event_order.flags.writeable)
        self.assertEqual(stock.nbytes, before_size)

    def test_raw_pit_projection_retains_gaps_nan_revisions_and_short_query_state(self):
        fields = ["P($$eps)", "$$eps", "PRef($$eps, -1)", "PRef($$eps, -3)",
                  "P(Mean($$eps, 3))", "P($$eps / $$roe)"]
        actual = self.provider.features([A], fields, self.dates[12], self.dates[16], allow_future=False)
        expected = self.full_history(self.provider, fields, self.dates[12], self.dates[16], allow_future=False)
        self.assert_same(actual, expected)
        for date in self.dates[:16]:
            cutoff = date.year * 10000 + date.month * 100 + date.day
            visible = self.rows[(self.rows.field == "eps") & (self.rows.date <= cutoff)]
            visible = visible.sort_values("date").drop_duplicates("period", keep="last").sort_values("period")
            fields_raw = fields[:4]
            values = self.provider.features([A], fields_raw, date, date, allow_future=False).iloc[0]
            latest = np.nan if visible.empty else visible.value.iloc[-1]
            np.testing.assert_allclose(values.iloc[:2], [latest, latest], equal_nan=True)
            if not visible.empty:
                first_period, last_period = int(visible.period.min()), int(visible.period.max())
                first_ordinal = first_period // 100 * 4 + first_period % 100 - 1
                last_ordinal = last_period // 100 * 4 + last_period % 100 - 1
                table = dict(zip(visible.period, visible.value))
                for column, offset in ((2, -1), (3, -3)):
                    ordinal = last_ordinal + offset
                    period = ordinal // 4 * 100 + ordinal % 4 + 1
                    expected_value = table.get(period, np.nan) if ordinal >= first_ordinal else np.nan
                    np.testing.assert_allclose(values.iloc[column], expected_value, equal_nan=True)

    def test_linear_pit_order_validation_rejects_checksum_valid_unsorted_data(self):
        path = self.financial / A / "pit.data"
        original = path.read_bytes()
        header = list(HEADER.unpack(original[:HEADER.size]))
        records = np.frombuffer(original[HEADER.size:], dtype=RECORD_DTYPE).copy()
        same_field = np.flatnonzero(records["field_id"][1:] == records["field_id"][:-1])
        same_period = np.flatnonzero((records["field_id"][1:] == records["field_id"][:-1]) &
                                    (records["period"][1:] == records["period"][:-1]))
        pairs = [(0, len(records) - 1), (int(same_field[0]), int(same_field[0]) + 1),
                 (int(same_period[0]), int(same_period[0]) + 1)]
        for left, right in pairs:
            damaged = records.copy()
            damaged[[left, right]] = damaged[[right, left]]
            payload = damaged.tobytes()
            header[-1] = hashlib.sha256(payload).digest()
            path.write_bytes(HEADER.pack(*header) + payload)
            with self.assertRaisesRegex(ValueError, "Unsorted"):
                StockPIT.read(path.parent)
        path.write_bytes(original)

    def test_registry_name_maps_refresh_and_preserve_suffix_ambiguity(self):
        store = self.provider._pit
        self.assertEqual(store.resolve("$$asset"), self.ids["other.asset"])
        register_fields(self.financial, {"balance.asset": {"source": "fina_indicator_vip"}})
        with self.assertRaises(KeyError):
            store.resolve("asset")
        self.assertEqual(store.resolve("other.asset"), self.ids["other.asset"])
        self.assertNotEqual(store.resolve("balance.asset"), self.ids["other.asset"])
        store.clear()
        self.assertEqual(store.resolve("eps"), self.ids["eps"])


if __name__ == "__main__":
    unittest.main()
