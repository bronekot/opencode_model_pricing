from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import opencode_pricing as pricing

FIXTURES = Path(__file__).parent / 'fixtures'


class PricingTests(unittest.TestCase):
    def test_matching_preserves_versions_and_variants(self):
        fixture = json.loads((FIXTURES / 'matching.json').read_text())
        for name, expected in fixture['cases']:
            with self.subTest(name=name):
                result = pricing.match_model_to_benchmarks(name, fixture['benchmarks'])
                self.assertEqual(result['source_model'] if result else None, expected)

    def test_table_headers_entities_and_missing_prices(self):
        models = pricing.parse_html_to_models((FIXTURES / 'pricing.html').read_text())
        self.assertEqual(len(models), 3)
        self.assertEqual(models[0], ['GPT 6 Sol (≤ 272K tokens)', '$2.00', '$10.00', '$0.20', '-'])
        self.assertEqual(models[1][0], 'Example Free')
        self.assertEqual(models[2][0], 'A & B (> 200K tokens)')
        self.assertIsNone(pricing.parse_html_to_models('<table><tr><td>Error</td></tr></table>'))

    def test_prices_and_metric_units(self):
        self.assertEqual(pricing.parse_price(' Free '), 0)
        self.assertEqual(pricing.parse_price('$1,234.50'), 1234.5)
        self.assertEqual(pricing.parse_price('$broken'), float('inf'))
        self.assertEqual(pricing.format_benchmark(1, percentage=True), '100.0%')
        self.assertEqual(pricing.format_benchmark(0.5), '0.5')
        self.assertEqual(pricing.compute_cod_index(48.1), 100)
        self.assertGreater(pricing.compute_cod_index(70), 100)

    def test_bridgebench_does_not_mix_versions(self):
        self.assertEqual(pricing.find_bridgebench('GPT 6 Sol (> 272K tokens)'), 643)
        self.assertEqual(pricing.find_bridgebench('GLM 5.3 Flash'), 450)
        self.assertIsNone(pricing.find_bridgebench('Claude Opus 4.6'))
        self.assertIsNone(pricing.find_bridgebench('Qwen3.8 Flash'))

    def test_environment_precedence_and_legacy_key(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / '.ENV').write_text('ARTIFICICAL_ANALYSIS_API="legacy"\n')
            with patch.object(pricing, 'BASE_DIR', base), patch.dict('os.environ', {}, clear=True):
                self.assertEqual(pricing.get_api_key(), 'legacy')
                with patch.dict('os.environ', {'ARTIFICIAL_ANALYSIS_API': 'environment'}):
                    self.assertEqual(pricing.get_api_key(), 'environment')

    def test_cache_refresh_failure_offline_and_migration(self):
        fixture = json.loads((FIXTURES / 'matching.json').read_text())['benchmarks']
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch.object(pricing, 'BASE_DIR', Path(directory)), patch.object(pricing, 'fetch_benchmarks', return_value=fixture) as fetch:
                result = pricing.get_benchmarks_for_models(['GPT 6 Sol'], 'test-key')
                self.assertEqual(result['GPT 6 Sol']['coding'], 40)
                fetch.assert_called_once()
                pricing.get_benchmarks_for_models(['Unknown Model'], 'test-key')
                fetch.assert_called_once()  # Новая неизвестная модель не обходит общий TTL.
                cache = pricing.load_benchmark_cache()
                cache['fetched_at'] -= pricing.BENCHMARK_TTL + 1
                pricing.save_benchmark_cache(cache)
                pricing.get_benchmarks_for_models(['GPT 6 Sol'], 'test-key', offline=True)
                fetch.assert_called_once()
                updated = {'GPT-6 Sol (max)': {'gpqa': 1, 'coding': 55}}
                fetch.return_value = updated
                self.assertEqual(pricing.get_benchmarks_for_models(['GPT 6 Sol'], 'test-key')['GPT 6 Sol']['coding'], 55)
                fetch.return_value = None
                self.assertEqual(pricing.get_benchmarks_for_models(['GPT 6 Sol'], 'test-key', refresh=True)['GPT 6 Sol']['coding'], 55)
                self.assertEqual(pricing.load_benchmark_cache()['models'], updated)
                Path(pricing.get_benchmark_cache_path()).write_text(json.dumps(fixture))
                self.assertEqual(pricing.load_benchmark_cache(), {})
                Path(pricing.get_benchmark_cache_path()).write_text('{broken')
                self.assertEqual(pricing.load_benchmark_cache(), {})

    def test_offline_cli_never_uses_network_and_formats_nulls(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pricing, 'BASE_DIR', Path(directory)):
            pricing.save_models_to_cache([['GPT 6 Sol', '-', '$10.00'], ['Example Free', 'Free', 'Free']])
            with patch.object(pricing, 'fetch_html_from_website', side_effect=AssertionError('network')), patch.object(pricing, 'fetch_benchmarks', side_effect=AssertionError('network')):
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(pricing.main(['--offline']), 0)
                self.assertIn('643.0', output.getvalue())
                self.assertIn('$0.0000', output.getvalue())
                self.assertNotIn('$inf', output.getvalue())

    def test_missing_prices_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pricing, 'BASE_DIR', Path(directory)), redirect_stdout(io.StringIO()):
            self.assertEqual(pricing.main(['--offline']), 1)


if __name__ == '__main__':
    unittest.main()
