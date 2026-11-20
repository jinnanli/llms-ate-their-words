"""CPU-only evaluator regressions using small trusted synthetic programs.

Run from the repository root. --compare-index additionally checks the staged
pre-change evaluator during development; the normal suite needs no Git history.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[2]
PATH = 'code/evaluation/tasks/code/eval_code.py'
spec = importlib.util.spec_from_file_location('code_evaluator', ROOT / PATH)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

CASES = [
    ('exact', "print(input())", ['hello'], ['hello'], [True]),
    ('multiple_inputs', "print(input())", ['one', 'two'], ['one', 'two'], [True, True]),
    ('whitespace', "print('  a  '); print(' b ')", [''], ['a\nb'], [True]),
    ('blank_line', "print('a'); print(); print('b')", [''], ['a\nb'], [-2]),
    ('decimal', "print('1.00 2.0')", [''], ['1 2'], [True]),
    ('empty', "pass", [''], [''], [True]),
    ('write', "sys.stdout.write('a'); sys.stdout.write('b')", [''], ['ab'], [True]),
    ('wrong_value', "print('wrong')", [''], ['right'], [-2]),
    ('wrong_length', "print('a'); print('b')", [''], ['a'], [-2]),
    ('runtime_error', "print('partial'); raise ValueError('synthetic')", [''], ['partial'], [-4]),
    ('system_exit', "print('ok'); raise SystemExit(0)", [''], ['ok'], [True]),
    ('close_legacy_argument', "print('ok'); sys.stdout.close(None); print('again')", [''], ['ok\nagain'], [True]),
    ('close_no_argument', "sys.stdout.close()", [''], [''], [-4]),
]

def run_case(module, case):
    _, code, inputs, outputs, _ = case
    stdout = sys.stdout
    try:
        result = module.grade_stdio(code, inputs, outputs, timeout=3)
        assert sys.stdout is stdout, 'stdout not restored'
        return result
    finally:
        module.disarm_timeout()

class CaptureRegression(unittest.TestCase):
    def test_stdio_scores_and_stdout_restoration(self):
        for case in CASES:
            with self.subTest(case=case[0]):
                scores, metadata = run_case(evaluator, case)
                self.assertEqual(scores, case[-1])
                if scores[-1] in [-2, -4]:
                    self.assertEqual(metadata['error_code'], scores[-1])

    def test_timeout_classification(self):
        from unittest.mock import patch
        stdout = sys.stdout
        with patch.object(evaluator, 'call_method', side_effect=evaluator.TimeoutException('synthetic')):
            scores, metadata = evaluator.grade_stdio('pass', [''], [''], 3)
        self.assertEqual(scores, [-3])
        self.assertEqual(metadata['error_message'], 'Time Limit Exceeded')
        self.assertIs(sys.stdout, stdout)

    def test_base_exception_restores_stdout(self):
        from unittest.mock import patch
        stdout = sys.stdout
        with patch.object(evaluator, 'call_method', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                evaluator.grade_stdio('pass', [''], [''], 3)
        self.assertIs(sys.stdout, stdout)

if __name__ == '__main__':
    if '--compare-index' in sys.argv:
        sys.argv.remove('--compare-index')
        old = types.ModuleType('previous_evaluator')
        source = subprocess.check_output(['git', 'show', ':' + PATH], cwd=ROOT)
        exec(compile(source.decode('utf-8-sig'), '<staged evaluator>', 'exec'), old.__dict__)
        for case in CASES:
            before, after = run_case(old, case), run_case(evaluator, case)
            assert before[0] == after[0], case[0]
            for key in ['error_code', 'error_message', 'output', 'inputs', 'expected']:
                assert before[1].get(key) == after[1].get(key), (case[0], key)
        print('STAGED_BASELINE_COMPARISON: PASS (13 synthetic cases)')
    unittest.main()
