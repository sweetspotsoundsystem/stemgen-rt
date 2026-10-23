#!/usr/bin/env python3
"""Compare callback and worker scheduling with the unchanged released graph."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILTER = 'RealtimeStemSanityTest.DISABLED_StemsAreNotAllIdenticalWhenAsyncRealtimePaced'
CASES = {
    'callback_qos_worker_qos': (0, 0),
    'callback_qos_worker_rt': (0, 1),
    'callback_rt_worker_qos': (1, 0),
    'callback_rt_worker_rt': (1, 1),
}
NAMES = list(CASES)
ORDER = [[NAMES[i] for i in row] for row in ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))]
ENDURANCE = NAMES[2:]
LONG_SOAK_CALLBACKS = (1800 * 44100 + 127) // 128


def trial_plan(protocol):
    """Return the complete exposure before any build or measurement starts."""
    def item(name, case, count, traced=False):
        return dict(name=name, case=case, measured_callbacks=count, traced=traced)
    if protocol == 'comparison':
        planned = [item('round' + str(index) + '-' + case, case, 10000)
                   for index, order in enumerate(ORDER, 1) for case in order]
        planned += [item('endurance-' + case, case, 105000) for case in ENDURANCE]
        planned += [item('trace-' + case, case, 10000, True) for case in ENDURANCE]
        return planned
    if protocol == 'long-soak':
        return [item('soak1-callback_rt_worker_rt', NAMES[3], LONG_SOAK_CALLBACKS),
                item('control-callback_rt_worker_qos', NAMES[2], LONG_SOAK_CALLBACKS),
                item('soak2-callback_rt_worker_rt', NAMES[3], LONG_SOAK_CALLBACKS)]
    raise ValueError('Unknown scheduling protocol: ' + protocol)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def capture(command, optional=False):
    result = subprocess.run(command, text=True, capture_output=True, timeout=30, cwd=ROOT)
    if not optional:
        result.check_returncode()
    return result.stdout.strip() if result.returncode == 0 else 'unavailable: ' + result.stderr.strip()


def source_inventory():
    manifest = ROOT / 'SOURCE_MANIFEST.json'
    if manifest.is_file():
        declared = json.loads(manifest.read_text())['files']
        names = sorted(declared)
    else:
        declared = None
        names = sorted(set(capture(['git', 'ls-files', '-co', '--exclude-standard', '-z']).split('\0')) - {''})
    result = {}
    for name in names:
        path = ROOT / name
        resolved=path.resolve()
        require(resolved.is_relative_to(ROOT) and resolved.is_file()
                and (declared is None or resolved==path),
                'Source escapes checkout or sealed source contains an alias: ' + name)
        result[name] = sha(path)
    if declared is not None:
        require(result == declared, 'Sealed plugin source changed')
        result['SOURCE_MANIFEST.json'] = sha(manifest)
    return result


def parse_summary(log, case, callbacks, traced):
    lines = [line for line in log.splitlines() if line.startswith('STEMGENRT_QUALIFICATION_SUMMARY ')]
    require(len(lines) == 1, 'Missing or repeated callback summary')
    pairs = [item.split('=', 1) for item in lines[0].split()[1:]]
    row = dict(pairs)
    require(len(row) == len(pairs), 'Duplicate summary field')
    callback_rt, worker_rt = CASES[case]
    expected = dict(warmup_callbacks='100', measured_callbacks=str(callbacks),
        worker_trace='enabled' if traced else 'disabled', ort_intra_op_threads_override='1',
        ort_intra_op_threads='1', callback_samples='128', sample_rate='44100', pdc_samples='256',
        worker_scheduling='realtime' if worker_rt else 'qos',
        callback_scheduling='realtime' if callback_rt else 'qos',
        worker_computation_ns='2700000', callback_computation_ns='500000', callback_constraint_ns='1000000',
        worker_policy_initial='applied', worker_policy_final='applied',
        callback_policy_initial='applied', callback_policy_final='applied',
        callback_priority='applied', worker_priority='applied')
    require(all(row.get(key) == value for key, value in expected.items()),
            'Callback/worker configuration, policy readback or exposure differs')
    require(row.get('status') in ('pass', 'fail'), 'Unknown callback result')
    # Independently forbid a pass with any failure counter. Detailed numerical
    # correctness is also required by the complete C++ suite before trials.
    zeros = ('deadline_misses', 'start_deadline_misses', 'start_catchup_intervals',
        'due_boundary_misses', 'last_wait_us', 'max_wait_us', 'wait_budget_us',
        'underrun_samples_last_block', 'underrun_samples', 'underrun_blocks',
        'underrun_active', 'queue_full_drops', 'ring_overflow_events',
        'ring_overflow_samples', 'unsafe_realtime_current', 'unsafe_realtime_callbacks')
    require(all(key in row and row[key].isdigit() for key in zeros), 'Missing or invalid failure counter')
    if row['status'] == 'pass':
        require(all(int(row[key]) == 0 for key in zeros) and row.get('finite_outputs') == '1'
                and float(row['reconstruction_max_abs']) <= 1e-6
                and float(row['max_us']) < 128000000 / 44100
                and float(row['p99.9_us']) < 128000000 / 44100
                and all(float(row[key])>1e-3 for key in ('drums_max_abs','bass_max_abs','vocals_max_abs'))
                and all(float(row[key])>1e-5 for key in ('drums_bass_max_abs_diff',
                    'drums_vocals_max_abs_diff','bass_vocals_max_abs_diff')),
                'Passing callback summary contains failures')
    return row


def run(args, output):
    chip = capture(['sysctl', '-n', 'machdep.cpu.brand_string'])
    require(chip in ('Apple M4', 'Apple M4 Pro'), 'Use a physical Apple M4 or M4 Pro')
    before = source_inventory()
    planned = trial_plan(args.protocol)
    # Build a fresh, explicit Release configuration. These overrides belong to
    # other builds and can otherwise change compiler/runtime identity silently.
    overrides=('CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS','OBJCFLAGS','ARCHFLAGS',
        'SDKROOT','DEVELOPER_DIR','MACOSX_DEPLOYMENT_TARGET','CMAKE_TOOLCHAIN_FILE',
        'CMAKE_PREFIX_PATH','CMAKE_INCLUDE_PATH','CMAKE_LIBRARY_PATH','CPM_SOURCE_CACHE',
        'CPATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','DYLD_INSERT_LIBRARIES','DYLD_LIBRARY_PATH')
    removed=sorted({key for key in overrides if key in os.environ}
                   | {key for key in os.environ if key.startswith('DYLD_')})
    for key in removed:
        os.environ.pop(key)
    write(output / 'plan.json', dict(schema='stemgenrt-plugin-scheduling-v2', frozen_utc=utc(),
        chip=chip, machine=platform.machine(), macos=capture(['sw_vers']), power_notes=args.power_notes,
        compiler=capture(['xcrun','clang++','--version']), cmake=capture(['cmake','--version']),
        ignored_build_override_names=removed,
        host={key:capture(['sysctl','-n',key],optional=True) for key in
            ('hw.model','hw.memsize','hw.physicalcpu','hw.perflevel0.physicalcpu','hw.perflevel1.physicalcpu')},
        source_files=before, cases=CASES, protocol=args.protocol, planned_trials=planned,
        warmup_callbacks=100, promotion_automatic=False, daw_qualified=False))
    write(output / 'power-before.json', dict(source=capture(['pmset', '-g', 'batt']),
        settings=capture(['pmset', '-g', 'custom']), thermal=capture(['pmset', '-g', 'therm'], optional=True)))

    def stage(name, command, timeout=900, env=None, allow_timing_failure=False):
        started = utc()
        with (output / (name + '.log')).open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True, timeout=timeout)
        write(output / (name + '-execution.json'), dict(command=command, started_utc=started,
            ended_utc=utc(), exit_code=result.returncode))
        require(result.returncode == 0 or (allow_timing_failure and result.returncode == 1),
                'Stage failed: ' + name)
        return result.returncode

    build = output / 'build'
    command = ['cmake', '-S', str(ROOT), '-B', str(build), '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_ARCHITECTURES=arm64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0',
        '-DSTEMGENRT_REQUIRE_QUALIFIED_ORT=ON']
    for name in ('juce', 'googletest'):
        if (ROOT / 'libs' / name / 'CMakeLists.txt').is_file():
            command.append('-DCPM_' + name.upper() + '_SOURCE=' + str(ROOT / 'libs' / name))
    stage('configure', command)
    stage('build', ['cmake', '--build', str(build), '--target', 'AudioPluginTest_BundleModel',
        'RealtimeSafetyTest', '--parallel', '2'], timeout=1800)
    shutil.copyfile(build/'CMakeCache.txt',output/'CMakeCache.txt')
    stage('correctness', ['ctest', '--test-dir', str(build), '--output-on-failure'], timeout=900)
    binary = build / 'test/AudioPluginTest'
    library = ROOT / 'libs/onnxruntime/lib/libonnxruntime.1.26.0.dylib'
    identities = {str(path):sha(path) for path in (binary, ROOT / 'model/model.onnx', library)}
    # Loader logging belongs only to this untimed numerical preflight.
    stage('loaded-runtime', [str(binary), '--gtest_color=no', '--gtest_filter=CroppedModelParityTest.*'],
        env=dict(os.environ, DYLD_PRINT_LIBRARIES='1'))
    loaded=set()
    for line in (output/'loaded-runtime.log').read_text().splitlines():
        if line.startswith('dyld') and 'libonnxruntime' in line and '/' in line:
            loaded.add(Path(line[line.index('/'):].strip()).resolve(strict=True))
    require(len(loaded)==1 and all(path.parent==library.parent and sha(path)==sha(library) for path in loaded),
            'The numerical preflight did not load exactly the authenticated SDK runtime')
    write(output / 'binary-and-runtime.json', dict(sha256=identities,
        preflight_loaded_runtime=[str(path) for path in sorted(loaded)],
        dependencies=capture(['otool', '-L', str(binary)])))
    results = []

    def trial(name, case, count, traced=False):
        print('Running ' + name + ': ' + str(count) + ' measured callbacks', flush=True)
        callback_rt, worker_rt = CASES[case]
        env = dict(os.environ, STEMGENRT_PACED_CALLBACK_REALTIME=str(callback_rt),
            STEMGENRT_PACED_WORKER_REALTIME=str(worker_rt), STEMGENRT_TRACE_WORKER='1' if traced else '0',
            STEMGENRT_PACED_ORT_THREADS='1', STEMGENRT_QUALIFICATION_CALLBACKS=str(count))
        # Never inherit another run's diagnostic destination.
        env.pop('STEMGENRT_PACED_TRACE_PATH', None)
        if traced:
            env['STEMGENRT_PACED_TRACE_PATH'] = str(output / name)
        if args.protocol == 'long-soak':
            write(output / (name + '-power-before.json'), dict(
                source=capture(['pmset', '-g', 'batt']),
                thermal=capture(['pmset', '-g', 'therm'], optional=True)))
        code = stage(name, [str(binary), '--gtest_also_run_disabled_tests', '--gtest_color=no',
            '--gtest_filter=' + FILTER], timeout=60 + 3 * count * 128 / 44100,
            env=env, allow_timing_failure=True)
        row = parse_summary((output / (name + '.log')).read_text(), case, count, traced)
        require((row['status'] == 'pass') == (code == 0), 'Exit code and qualification gate disagree')
        if traced:
            with (output / (name + '-callbacks.csv')).open(newline='') as stream:
                callbacks = list(csv.DictReader(stream))
            require(len(callbacks)==100+count and all(int(r['callback'])==i for i,r in enumerate(callbacks)),
                    'Incomplete callback trace')
            require((output / (name + '-worker.csv')).stat().st_size>100, 'Missing worker trace')
        results.append(dict(name=name, case=case, traced=traced, exit_code=code, summary=row))
        write(output / 'partial-results.json', results)
        if args.protocol == 'long-soak':
            write(output / (name + '-power-after.json'), dict(
                source=capture(['pmset', '-g', 'batt']),
                thermal=capture(['pmset', '-g', 'therm'], optional=True)))

    for item in planned:
        trial(item['name'], item['case'], item['measured_callbacks'], item['traced'])
    require(source_inventory() == before, 'Source changed during comparison')
    require(all(sha(path) == digest for path, digest in identities.items()), 'Binary/model/runtime changed')
    write(output / 'power-after.json', dict(source=capture(['pmset', '-g', 'batt']),
        thermal=capture(['pmset', '-g', 'therm'], optional=True)))
    candidates = [row for row in results if row['case'] == NAMES[3] and not row['traced']]
    interpretation = ('Two separate thirty-minute real-time-worker trials bracket one thirty-minute '
        'QoS-worker control. The synthetic callback keeps real-time scheduling throughout, and tracing '
        'is disabled. Power/thermal snapshots surround each trial. Sequential order cannot eliminate '
        'all thermal/time confounding. Installed DAW behavior is not measured.'
        if args.protocol == 'long-soak' else
        'Four order-balanced rounds of the complete callback/worker path; two separate '
        'five-minute endurance trials hold the synthetic callback at real-time policy. '
        'Worker trace diagnostics run last and are excluded from uninstrumented timing. '
        'Endurance order is sequential; installed DAW behavior is not measured.')
    write(output / 'results.json', dict(status='comparison_completed', chip=chip,
        protocol=args.protocol, planned_trials=planned, trials=results,
        all_timing_gates_passed=all(row['exit_code'] == 0 for row in results if not row['traced']),
        candidate_trials=len(candidates),
        candidate_all_timing_gates_passed=bool(candidates) and all(row['exit_code'] == 0 for row in candidates),
        daw_qualified=False, promotion_automatic=False,
        interpretation=interpretation + ' All original callback, underrun, reconstruction and zero-wait '
            'gates remain required. A completed comparison is not a production promotion.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--power-notes', required=True)
    parser.add_argument('--protocol', choices=('comparison', 'long-soak'), default='comparison',
        help='Original balanced comparison, or two 30-minute real-time-worker trials around a 30-minute QoS control')
    args = parser.parse_args()
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        parser.error('Use native arm64 Python on a physical target Mac')
    output = args.output.resolve()
    if output.suffix or output.exists() or output.with_suffix('.zip').exists() or output.is_relative_to(ROOT):
        parser.error('Choose a new directory outside the source tree, without a file extension')
    output.mkdir(parents=True)
    caffeine = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
    success = False
    try:
        run(args, output)
        success = True
    except (Exception, KeyboardInterrupt):
        (output / 'failure.txt').write_text(traceback.format_exc())
        print('Stopped; diagnostics: ' + str(output / 'failure.txt'), file=sys.stderr)
    finally:
        caffeine.terminate()
        caffeine.wait(timeout=10)
        files = sorted(path for path in output.iterdir() if path.is_file())
        write(output / 'result-files.json', {path.name:sha(path) for path in files})
        archive = output.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as target:
            for path in sorted(output.iterdir()):
                if path.is_file():
                    target.write(path, output.name + '/' + path.name)
        print('Results ZIP: ' + str(archive), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
