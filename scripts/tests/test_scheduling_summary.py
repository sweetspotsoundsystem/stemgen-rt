"""Reject misidentified or falsely passing scheduling evidence; no timing claims."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'compare-paced-scheduling-macos.py'
SPEC = importlib.util.spec_from_file_location('scheduling_probe', PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class SchedulingSummaryTest(unittest.TestCase):
    def setUp(self):
        self.case = 'callback_rt_worker_rt'
        self.row = dict(status='pass', warmup_callbacks='100', measured_callbacks='10000',
            worker_trace='disabled', ort_intra_op_threads_override='1', ort_intra_op_threads='1',
            callback_samples='128', sample_rate='44100', pdc_samples='256',
            worker_scheduling='realtime', callback_scheduling='realtime',
            worker_computation_ns='2700000', callback_computation_ns='500000', callback_constraint_ns='1000000',
            worker_policy_initial='applied', worker_policy_final='applied', callback_policy_initial='applied',
            callback_policy_final='applied', callback_priority='applied', worker_priority='applied',
            deadline_misses='0', start_deadline_misses='0', start_catchup_intervals='0', due_boundary_misses='0',
            last_wait_us='0', max_wait_us='0', wait_budget_us='0', underrun_samples_last_block='0',
            underrun_samples='0', underrun_blocks='0', underrun_active='0', queue_full_drops='0',
            ring_overflow_events='0', ring_overflow_samples='0', unsafe_realtime_current='0',
            unsafe_realtime_callbacks='0', finite_outputs='1', reconstruction_max_abs='1e-7',
            max_us='50', **{'p99.9_us':'40'}, drums_max_abs='.1', bass_max_abs='.1', vocals_max_abs='.1',
            drums_bass_max_abs_diff='.01', drums_vocals_max_abs_diff='.01', bass_vocals_max_abs_diff='.01')

    def log(self):
        return 'STEMGENRT_QUALIFICATION_SUMMARY ' + ' '.join(key+'='+value for key,value in self.row.items())

    def parse(self):
        return PROBE.parse_summary(self.log(), self.case, 10000, False)

    def test_complete_synthetic_pass(self):
        self.assertEqual(self.parse(), self.row)

    def test_measured_deadline_failure_is_retained(self):
        self.row.update(status='fail', underrun_samples='128', due_boundary_misses='1')
        self.assertEqual(self.parse()['status'], 'fail')

    def test_failure_cannot_be_labeled_pass(self):
        for key,value in [('underrun_samples','1'), ('start_catchup_intervals','1'),
                          ('drums_max_abs','0'), ('max_us','4000'), ('reconstruction_max_abs','nan')]:
            with self.subTest(key=key):
                old=self.row[key]; self.row[key]=value
                with self.assertRaises(ValueError): self.parse()
                self.row[key]=old

    def test_wrong_or_demoted_policy_is_rejected(self):
        for key,value in [('worker_scheduling','qos'), ('callback_scheduling','qos'),
                          ('worker_policy_final','failed'), ('callback_policy_initial','failed'),
                          ('worker_computation_ns','2000000'), ('pdc_samples','384'),
                          ('ort_intra_op_threads','2'), ('worker_trace','enabled')]:
            with self.subTest(key=key):
                old=self.row[key]; self.row[key]=value
                with self.assertRaises(ValueError): self.parse()
                self.row[key]=old

    def test_missing_and_repeated_summaries_are_rejected(self):
        for log in ('',self.log()+'\n'+self.log(),self.log()+' status=pass'):
            with self.subTest(log=log[:30]):
                with self.assertRaises(ValueError):
                    PROBE.parse_summary(log,self.case,10000,False)

    def test_each_condition_occupies_each_short_trial_position(self):
        for position in range(4):
            self.assertEqual({row[position] for row in PROBE.ORDER},set(PROBE.CASES))
        for row in PROBE.ORDER:
            self.assertEqual(set(row),set(PROBE.CASES))


if __name__ == '__main__':
    unittest.main()
