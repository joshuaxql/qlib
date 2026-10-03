"""Public export isolation, offline resources, hostile labels and asset integrity."""

from hashlib import sha256
from html.parser import HTMLParser
from importlib.resources import files
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.contrib.report.analysis_model import analyze_factors
from qlib.contrib.report.analysis_model._factor_report import _assets


class _Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.payload = ''
        self.external = []
        self.capture = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script':
            self.capture = attrs.get('id') == 'report-data'
            if 'src' in attrs:
                self.external.append(attrs['src'])
        if tag == 'link' and attrs.get('rel') == 'stylesheet':
            self.external.append(attrs.get('href'))

    def handle_endtag(self, tag):
        if tag == 'script':
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.payload += data


class _CalendarProvider:
    def __init__(self, dates):
        self.dates = pd.DatetimeIndex(dates)

    def calendar(self):
        return self.dates


class _NamedIndustryProvider(_CalendarProvider):
    def __init__(self, dates):
        super().__init__(dates)
        self.membership_calls = []
        self.name_calls = 0

    def industries(self):
        return ['801001', '801002']

    def industry_names(self):
        self.name_calls += 1
        return {'801001': '同名行业', '801002': '同名行业'}

    def universe(self, market, start, end):
        self.membership_calls.append(market)
        mask = pd.DataFrame(False, index=self.dates, columns=list('ABCD'))
        mask.loc[:, list('AB' if market == 'industry/801001' else 'CD')] = True
        return mask

    def stock_basic(self):
        raise AssertionError('Industry names must not replace historical code memberships')


class FactorHtmlExportTest(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range('2025-01-02', periods=5)
        index = pd.MultiIndex.from_product([list('ABCD'), dates], names=['instrument', 'datetime'])
        values = pd.DataFrame({'alpha': np.repeat([1., 2., 3., 4.], len(dates))}, index=index)
        labels = pd.DataFrame({1: values.alpha.to_numpy() / 100,
                               5: -values.alpha.to_numpy() / 100}, index=index)
        self.values, self.labels = values, labels
        self.dates = dates
        self.result = analyze_factors(values, labels, quantiles=2)

    def export_payload(self, result, provider):
        with TemporaryDirectory() as temporary:
            path = result.to_html(Path(temporary) / 'report.html', provider=provider)
            return json.loads(_Document(path.read_text(encoding='utf-8')).payload)

    def test_pending_forward_returns_require_calendar_endpoint_and_no_valid_labels(self):
        labels, values = self.labels.copy(), self.values.copy()
        labels.loc[(slice(None), self.dates[[0, 2, 3, 4]]), 1] = np.nan
        # A finite external zero return remains available despite absent factor
        # data: label availability must not be inferred from pair_count.
        labels.loc[('A', self.dates[-1]), 1] = 0.
        values.loc[('A', self.dates[-1]), 'alpha'] = np.nan
        labels[5] = np.nan
        labels.loc[('A', self.dates[1]), 5] = .005
        result = analyze_factors(values, labels, quantiles=2)
        result.config['entry_lag'] = 1
        before = {name: getattr(result, name).copy(deep=True) for name in (
            'factors', 'forward_returns', 'summary', 'daily', 'quantile_returns',
            'quantile_membership', 'turnover', 'autocorrelation')}
        config = result.config.copy()
        payload = self.export_payload(result, _CalendarProvider(self.dates))
        self.assertEqual(payload['meta']['price_data_end'], '2025-01-08')
        first = payload['factors'][0]['horizons']['1']['daily']
        self.assertEqual(first['pending_forward_return'], [False, False, False, True, False])
        self.assertEqual(first['label_count'], [0, 4, 0, 0, 1])
        self.assertEqual(first['pair_count'], [0, 4, 0, 0, 0])
        self.assertEqual(first['missing_label_count'], [4, 0, 4, 4, 3])
        fifth = payload['factors'][0]['horizons']['5']['daily']
        self.assertEqual(fifth['pending_forward_return'], [True, False, True, True, True])
        self.assertEqual(fifth['label_count'], [0, 1, 0, 0, 0])
        for name, table in before.items():
            pd.testing.assert_frame_equal(getattr(result, name), table, check_exact=True)
        self.assertEqual(result.config, config)

    def test_unknown_calendar_dates_are_unknown_and_valid_external_labels_are_not_pending(self):
        labels = self.labels * np.nan
        labels.loc[('A', self.dates[-1]), 1] = .123456789012345
        result = analyze_factors(self.values, labels, quantiles=2)
        result.config['entry_lag'] = 1
        calendar = self.dates.delete(2)
        payload = self.export_payload(result, _CalendarProvider(calendar))
        first = payload['factors'][0]['horizons']['1']['daily']
        self.assertEqual(first['pending_forward_return'], [False, False, None, True, False])
        self.assertIsNone(payload['factors'][0]['horizons']['5']['daily']['pending_forward_return'][2])
        self.assertEqual(first['label_count'][-1], 1)
        empty = self.export_payload(result, _CalendarProvider([]))
        self.assertNotIn('price_data_end', empty['meta'])
        for horizon in empty['factors'][0]['horizons'].values():
            self.assertNotIn('pending_forward_return', horizon['daily'])

    def test_duplicate_chinese_industry_names_do_not_merge_distinct_codes(self):
        labels = self.labels.copy()
        labels.loc[('C', slice(None)), 1] = .04
        labels.loc[('D', slice(None)), 1] = .03
        result = analyze_factors(self.values, labels, quantiles=2)
        provider = _NamedIndustryProvider(self.dates)
        payload = self.export_payload(result, provider)
        self.assertEqual(provider.name_calls, 1)
        self.assertEqual(provider.membership_calls, ['industry/801001', 'industry/801002'])
        horizon = payload['factors'][0]['horizons']['1']
        sector = horizon['sector']
        self.assertEqual([row['code'] for row in sector['overview']], ['801001', '801002'])
        self.assertEqual([row['name'] for row in sector['overview']], ['同名行业', '同名行业'])
        np.testing.assert_allclose([row['ic_mean'] for row in sector['overview']], [1., -1.])
        np.testing.assert_allclose([row['rank_ic_mean'] for row in sector['overview']], [1., -1.])
        first, second = sector['quantile_returns']
        self.assertEqual((first['code'], second['code']), ('801001', '801002'))
        self.assertAlmostEqual(first['groups'][0]['mean_return_bps'], 150.)
        self.assertIsNone(first['groups'][1]['mean_return_bps'])
        self.assertIsNone(second['groups'][0]['mean_return_bps'])
        self.assertAlmostEqual(second['groups'][1]['mean_return_bps'], 350.)
        options = horizon['chart_options']['common']
        self.assertEqual(options['sector-ic']['xAxis'][0]['data'], ['同名行业', '同名行业'])
        np.testing.assert_allclose(options['sector-ic']['series'][0]['data'], [1., -1.])
        self.assertIn('sector-0', options)
        self.assertIn('sector-1', options)
        # Fee switching must retain distinct industry codes even when their
        # display names match, and must preserve unavailable group returns.
        charged = horizon['fee_scenarios']['commission_stamp']
        net_rows = charged['sector']['quantile_returns']
        self.assertEqual([row['code'] for row in net_rows], ['801001', '801002'])
        self.assertEqual([row['name'] for row in net_rows], ['同名行业', '同名行业'])
        self.assertIsNone(net_rows[0]['groups'][1]['mean_return_bps'])
        self.assertIsNone(net_rows[1]['groups'][0]['mean_return_bps'])
        expected = (1.015 * .9987 / 1.0003 - 1) * 10000
        self.assertAlmostEqual(net_rows[0]['groups'][0]['mean_return_bps'], expected, places=10)
        net_options = charged['chart_options']['common']
        self.assertAlmostEqual(net_options['sector-0']['series'][0]['data'][0]['value'], expected, places=10)
        self.assertEqual(payload['meta']['fee_options'][0]['value'], 'none')
        self.assertEqual(payload['meta']['fee_options'][1]['label'], '3‱佣金 + 1‰印花税')

    def test_hostile_labels_are_json_text_and_report_has_no_external_assets(self):
        label = '</script><script>alert(1)</script><img src=x onerror=alert(2)>__REPORT_DATA__\u2028\u2029'
        result = analyze_factors(self.values.rename(columns={'alpha': label}), self.labels, quantiles=2)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'nested/report.html'
            self.assertEqual(result.to_html(path, title=label, industries=dict.fromkeys(list('ABCD'), label)), path)
            document = path.read_text(encoding='utf-8')
            parsed = _Document(document)
            payload = json.loads(parsed.payload)
            self.assertEqual(payload['meta']['title'], label)
            self.assertEqual(payload['factors'][0]['name'], label)
            self.assertEqual(payload['factors'][0]['horizons']['1']['sector']['overview'][0]['name'], label)
            self.assertNotIn(label, document)
            self.assertNotIn('<', parsed.payload)
            self.assertFalse(parsed.external)
            self.assertIn('Apache ECharts', document)
            self.assertEqual(len(payload['meta']['licenses']), 2)

    def test_standalone_analysis_needs_no_global_provider_or_matplotlib(self):
        with TemporaryDirectory() as temporary, patch('qlib.data.LocalProvider', side_effect=AssertionError('Unexpected provider')):
            payload = json.loads(_Document(self.result.to_html(Path(temporary) / 'report.html').read_text(encoding='utf-8')).payload)
            self.assertFalse(payload['factors'][0]['horizons']['1']['sector']['available'])
            # Perfect constant IC across dates still exports a usable histogram
            # with positive bin widths, rather than a degenerate floating range.
            for horizon in payload['factors'][0]['horizons'].values():
                for metric in ('ic', 'rank_ic'):
                    values = horizon['daily'][metric]
                    self.assertEqual(len(set(values)), 1)
                    distribution = horizon[f'{metric}_distribution']
                    self.assertEqual(sum(row['count'] for row in distribution['histogram']), len(self.dates))
                    self.assertTrue(all(row['width'] > 0 for row in distribution['histogram']))
                    self.assertEqual([row[1] for row in distribution['qq']], sorted(values))
        with self.assertRaisesRegex(ValueError, 'industries or provider'):
            self.result.to_html('ignored.html', industries={}, provider=object())
        with self.assertRaisesRegex(ValueError, 'html must be a bool'):
            self.result.save('ignored', html=1)

    def test_vendored_library_matches_pinned_distribution(self):
        directory = files('qlib.contrib.report.analysis_model').joinpath('_assets')
        provenance = json.loads(directory.joinpath('ECHARTS-PROVENANCE.json').read_text(encoding='utf-8'))
        self.assertEqual(sha256(directory.joinpath('echarts.min.js').read_bytes()).hexdigest(), provenance['sha256'])
        self.assertEqual(sha256(directory.joinpath('ECHARTS-D3-LICENSE.txt').read_bytes()).hexdigest(), provenance['third_party_license_sha256'])
        self.assertTrue(all(value for value in _assets().values()))

    def test_result_remains_exportable_after_original_data_directory_is_moved(self):
        with TemporaryDirectory() as temporary:
            self.result._report_provider_uri = str(Path(temporary) / 'missing_data')
            with patch('qlib.data.LocalProvider', side_effect=AssertionError('Missing provider instantiated')):
                path = self.result.to_html(Path(temporary) / 'report.html')
            payload = json.loads(_Document(path.read_text(encoding='utf-8')).payload)
            self.assertFalse(payload['factors'][0]['horizons']['1']['sector']['available'])

    def test_batch_exports_isolated_factor_folders_and_safe_names(self):
        values = pd.concat([self.values.rename(columns={'alpha': name}) for name in
                            ('alpha', 'ALPHA', '../outside', 'CON')], axis=1)
        result = analyze_factors(values, self.labels, quantiles=2)
        before = result.factors.copy(deep=True)
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reports'
            result.save(root)
            folders = list(root.iterdir())
            self.assertEqual(len(folders), 4)
            self.assertEqual(len({folder.name.casefold() for folder in folders}), 4)
            self.assertFalse((Path(temporary) / 'outside').exists())
            self.assertTrue((root / 'alpha' / 'report.html').is_file())
            for folder in folders:
                self.assertTrue(folder.is_dir())
                self.assertEqual(len(list(folder.glob('*.csv'))), 8)
                data = pd.read_csv(folder / 'factors.csv', index_col=[0, 1])
                self.assertEqual(len(data.columns), 1)
                name = data.columns[0]
                summary = pd.read_csv(folder / 'summary.csv')
                self.assertEqual(summary['factor'].unique().tolist(), [name])
                # Report presentation changes do not remove the established
                # public analysis columns or alter the CSV export contract.
                self.assertIn('long_short_mean', summary.columns)
                self.assertIn('long_short_std', summary.columns)
                self.assertIn('long_short_return', pd.read_csv(folder / 'daily.csv').columns)
                payload = json.loads(_Document((folder / 'report.html').read_text(encoding='utf-8')).payload)
                self.assertEqual([item['name'] for item in payload['factors']], [name])
                self.assertEqual(payload['meta']['renderer'], 'pyecharts')
                self.assertTrue(payload['factors'][0]['chart_options']['mean-quantile']['series'])
                self.assertNotIn('returns_metrics', payload['factors'][0])
                self.assertNotIn('quantile_stats', payload['factors'][0])
                for horizon in payload['factors'][0]['horizons'].values():
                    self.assertNotIn('factor_weighted', horizon)
                    self.assertNotIn('spread', horizon)
                    self.assertFalse({'weighted-cumulative', 'quantile-spread', 'spread-cumulative'}
                                     .intersection(horizon['chart_options']['common']))
                    for metric in ('ic', 'rank_ic'):
                        self.assertIn('ic-cumulative', horizon['chart_options'][metric])
            with self.assertRaisesRegex(ValueError, 'Choose factor'):
                result.to_html(root / 'batch.html')
            single = result.to_html(root / 'single.html', factor='ALPHA')
            self.assertEqual(json.loads(_Document(single.read_text(encoding='utf-8')).payload)['factors'][0]['name'], 'ALPHA')
        pd.testing.assert_frame_equal(result.factors, before)


if __name__ == '__main__':
    unittest.main()
