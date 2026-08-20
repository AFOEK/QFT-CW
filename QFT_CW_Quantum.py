from __future__ import annotations

import os
# Keep Qiskit's own process parallelism off; StatePreparation uses one persistent pool below.
# Aer still uses its native thread/experiment parallelism.
os.environ.setdefault("QISKIT_PARALLEL", "FALSE")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, dct, idct, dst, idst

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, HGate, SwapGate, MCXGate, SdgGate
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_aer.noise import NoiseModel

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns


WINDOW_SIZE = 256
RETENTION_RATIOS = [1.00, 0.85, 0.75, 0.50, 0.45, 0.25, 0.10, 0.05, 0.02, 0.01]
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
CW_DATASET = SCRIPT_DIR / "cw_dataset"
CLEAN_DIR = CW_DATASET / "clean"
NOISY_DIR = CW_DATASET / "noisy"
ENVELOPE_DIR = CW_DATASET / "envelopes"
AUDIO_DIR = CW_DATASET / "audio"
AUDIO_CLEAN_DIR = AUDIO_DIR / "clean"
AUDIO_NOISY_DIR = AUDIO_DIR / "noisy"
METADATA_FILE = CW_DATASET / "metadata.csv"

QUANTUM_WPM = 20
QUANTUM_SNRS = [30, 10, 0, -10]
SCIPY_WORKERS = -1

Q_BATCH_SIZE = 32
QISKIT_STATEPREP_PROCS = min(12, os.cpu_count() or 1)
QISKIT_TRANSPILE_PROCS = 1
CPU_QSIM = AerSimulator(method="statevector", device="CPU", max_parallel_threads=0, max_parallel_experiments=0)
STATEPREP_EXECUTOR: ProcessPoolExecutor | None = None

MODE="quantum"
OUTPUT_DIR=SCRIPT_DIR/"cw_outputs"
QUANTUM_WPM=20
QUANTUM_SNRS=[30,10,0,-10]
QUANTUM_MESSAGE_COUNT=2
QISKIT_STATEPREP_PROCS=min(18,os.cpu_count() or 1)
QISKIT_TRANSPILE_PROCS=4
Q_BATCH_SIZE=32
SAVE_PLOTS=True
SAVE_RECONSTRUCTED=True
RUN_NOISE=True
NOISE_MAX_WINDOWS=None

SAVE_CIRCUIT_RESOURCES=True
SAVE_CIRCUIT_DRAWINGS=True
SAVE_DETAILED_CIRCUITS=False
RESOURCE_OPT_LEVEL=1
RESOURCE_BASIS=["rz","sx","x","cx"]


def stabilize_amplitudes(x, zero_tol=1e-13):
    a = np.asarray(x, dtype=np.complex128).copy()
    if not np.all(np.isfinite(a)):
        raise ValueError("Non-finite amplitudes")
    scale = np.max(np.abs(a))
    if scale <= 1e-15:
        return None
    a[np.abs(a) < zero_tol * scale] = 0.0
    a = np.asarray(np.real_if_close(a, tol=1000), dtype=np.complex128)
    norm = np.linalg.norm(a)
    if norm <= 1e-15:
        return None
    a /= norm
    a /= np.sqrt(np.vdot(a, a).real)
    return a


def synthesize_stateprep(amplitudes,max_perturbation=1e-12):
    original=np.asarray(amplitudes,dtype=np.complex128).copy()

    try:
        return StatePreparation(original,normalize=False).definition,0.0,"exact"
    except ValueError as e:
        if "Input matrix is not unitary" not in str(e): raise
        last_error=e

    for decimals in (15,14,13,12):
        stable=np.round(original.real,decimals)+1j*np.round(original.imag,decimals)
        norm=np.linalg.norm(stable)
        if norm<=1e-15: continue
        stable/=norm
        perturbation=float(np.linalg.norm(original-stable))
        if perturbation>max_perturbation: continue
        try:
            return StatePreparation(stable,normalize=False).definition,perturbation,f"round-{decimals}"
        except ValueError as e:
            if "Input matrix is not unitary" not in str(e): raise
            last_error=e

    zero_mask=np.abs(original)<1e-15
    zero_count=int(np.count_nonzero(zero_mask))

    if zero_count:
        eps=min(1e-14,max_perturbation/(4*np.sqrt(zero_count)))
        stable=original.copy()
        idx=np.flatnonzero(zero_mask)
        stable[idx]=eps*np.where(np.arange(len(idx))%2==0,1.0,-1.0)
        stable/=np.linalg.norm(stable)

        perturbation=float(np.linalg.norm(original-stable))

        if perturbation<=max_perturbation:
            try:
                return StatePreparation(stable,normalize=False).definition,perturbation,f"dezero-{eps:.1e}"
            except ValueError as e:
                if "Input matrix is not unitary" not in str(e): raise
                last_error=e

    raise last_error


def build_stateprep_item(item):
    i,x=item
    x=np.asarray(x,dtype=np.complex128).copy()
    norm=np.linalg.norm(x)

    if norm<=1e-15:
        return i,norm,None,0.0,None

    amplitudes=stabilize_amplitudes(x)
    if amplitudes is None:
        return i,norm,None,0.0,None

    try:
        prep_definition,perturbation,retry_decimals=synthesize_stateprep(amplitudes)
    except Exception as e:
        raise RuntimeError(
            f"StatePreparation failed window={i}, norm={norm:.17e}, "
            f"nonzero={np.count_nonzero(amplitudes)}/{len(amplitudes)}, "
            f"max_imag={np.max(np.abs(amplitudes.imag)):.3e}"
        ) from e

    return i,norm,prep_definition,perturbation,retry_decimals



def validate_window(x):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be 1-D")

    n = len(x)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Length must be a power of two, got {n}")

    return x

def prepare_amplitudes(x):
    x = validate_window(x)
    x = np.asarray(x, dtype=np.complex128)
    x[np.abs(x) < 1e-14] = 0.0
    norm = np.linalg.norm(x)
    if norm <= 1e-15: 
        return None, 0.0
    amplitudes = x / norm

    qc = QuantumCircuit(int(np.log2(len(x))))
    qc.append(StatePreparation(amplitudes, normalize=True), range(qc.num_qubits))
    return qc, norm

def run_transform(x, transform_circuit):
    prep, input_norm = prepare_amplitudes(x)
    if prep is None:
        return (np.zeros_like(x, dtype=np.complex128,), None,)

    if prep.num_qubits != transform_circuit.num_qubits:
        raise ValueError("State preparation and transform have different qubit counts.")

    prep.compose(transform_circuit, inplace=True,)
    state = Statevector.from_instruction(prep).data
    result = (np.asarray(state) * input_norm)
    return result, prep

def append_zero_controlled_h(qc, controls, target):
    if not controls:
        qc.h(target)
        return

    for q in controls:
        qc.x(q)

    gate = HGate().control(len(controls))
    qc.append(gate, [*controls, target])
    for q in controls:
        qc.x(q)

def append_zero_controlled_swap(qc, controls, q1, q2):
    if not controls:
        qc.swap(q1, q2)
        return

    for q in controls:
        qc.x(q)

    gate = SwapGate().control(len(controls))
    qc.append(gate, [*controls, q1, q2])
    for q in controls:
        qc.x(q)

def append_mcx(qc, controls, target):
    controls = list(controls)
    if len(controls) == 0:
        qc.x(target)
    elif len(controls) == 1:
        qc.cx(controls[0], target)
    else:
        qc.append(MCXGate(len(controls)), [*controls, target])

def append_ctrl_ones_complement(qc, data, ctrl):
    for q in data:
        qc.cx(ctrl, q)

def append_ctrl_increment(qc, data, ctrl):
    n = len(data)
    for i in range(n - 1, 0, -1):
        controls = [ctrl, *data[:i]]
        append_mcx(qc, controls, data[i])
    qc.cx(ctrl, data[0],)

def append_ctrl_decrement(qc, data, ctrl):
    n = len(data)
    qc.cx(ctrl, data[0],)
    for i in range(1, n):
        controls = [ctrl, *data[:i],]
        append_mcx(qc, controls, data[i])

def append_ctrl_twos_complement(qc, data, ctrl):
    append_ctrl_ones_complement(qc, data, ctrl)
    append_ctrl_increment(qc, data, ctrl)

def append_vn(qc, data, ctrl):
    # H ⊗ I
    qc.h(ctrl)
    # pi_1:
    # |1,x> -> |1, one's_complement(x)>
    append_ctrl_ones_complement(qc, data, ctrl)

def append_d1(qc, data, ctrl):
    n = len(data)
    N = 2 ** n
    theta = np.pi / (2 * N)
    # delta_2:
    # active when ctrl = 1
    # K_i = X L_i^dag X
    for i, q in enumerate(data):
        angle = (2 ** i) * theta
        qc.x(q)
        qc.cp(-angle, ctrl, q)
        qc.x(q)

    # delta_1:
    # active when ctrl = 0
    qc.x(ctrl)
    for i, q in enumerate(data):
        angle = (2 ** i) * theta
        qc.cp(angle, ctrl, q)

    qc.x(ctrl)
    # C = diag(1, conjugate(omega))
    qc.p(-theta, ctrl,)

def append_g(qc, data, ctrl):
    # Base B^T = S H
    qc.h(ctrl)
    qc.s(ctrl)
    # Convert condition:
    # data == 000...0
    # into
    # data == 111...1
    for q in data:
        qc.x(q)

    controls = list(data)
    # Controlled J = Sdg H Sdg
    qc.append(SdgGate().control(len(controls)), [*controls, ctrl])
    qc.append(HGate().control(len(controls)), [*controls, ctrl])
    qc.append(SdgGate().control(len(controls)), [*controls, ctrl])
    # Restore data
    for q in data:
        qc.x(q)

def append_un_dagger(qc, data, ctrl):
    # D1
    append_d1(qc, data, ctrl)
    # Conditional two's complement
    append_ctrl_twos_complement(qc, data, ctrl)
    # Conditional branch mixing
    append_g(qc, data, ctrl,)
    # Conditional decrement
    append_ctrl_decrement(qc, data, ctrl)

def run_branch_transform(x, transform):
    x = validate_window(x)
    x = np.asarray(x, dtype=np.complex128)
    x[np.abs(x) < 1e-14] = 0.0
    norm = np.linalg.norm(x)
    if norm <= 1e-15:
        return np.zeros_like(x, dtype=np.complex128), None, 0.0

    amplitudes = x / norm
    N = len(x)
    n = int(np.log2(N))

    qc = QuantumCircuit(n + 1)
    qc.append(StatePreparation(amplitudes, normalize=True), range(n))
    qc.compose(transform, qubits=range(n + 1), inplace=True)

    state = np.asarray(Statevector.from_instruction(qc).data)
    output_branch = state[:N]
    result = output_branch * norm
    leakage = max(0.0, 1.0 - np.sum(np.abs(output_branch) ** 2))
    #print(f"norm={np.linalg.norm(amplitudes):.17f} finite={np.all(np.isfinite(amplitudes))} zeros={np.sum(amplitudes == 0)}/{len(amplitudes)} max={np.max(np.abs(amplitudes)):.3e}")
    return result, qc, leakage

def append_qct_qst_core(qc, data, ctrl):
    # 1. V_N
    append_vn(qc, data, ctrl)
    # 2. QFT_{2N}
    qft = build_qft(len(data) + 1)
    qc.compose(qft, qubits=[*data, ctrl,], inplace=True)
    # 3. U_N^\dagger
    append_un_dagger(qc, data, ctrl,)

def resolve_dataset_path(file_value, default_dir):
    path = Path(file_value)
    if path.is_absolute():
        return path

    candidate = CW_DATASET / path
    if candidate.exists():
        return candidate

    candidate = default_dir / path.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not locate dataset file: {file_value}")

def load_cw_sample(row):
    clean_path = resolve_dataset_path(row["clean_file"], CLEAN_DIR)
    noisy_path = resolve_dataset_path(row["noisy_file"], NOISY_DIR)
    clean = np.load(clean_path).astype(np.float64)
    noisy = np.load(noisy_path).astype(np.float64)
    return {
        "clean": clean,
        "noisy": noisy,
        "clean_path": clean_path,
        "noisy_path": noisy_path,
        "metadata": row,
    }

def split_windows(signal, window_size=WINDOW_SIZE):
    signal = np.asarray(signal, dtype=np.float64,)
    original_length = len(signal)
    pad = (-original_length) % window_size
    if pad > 0:
        signal = np.pad(signal, (0, pad), mode="constant",)

    windows = signal.reshape(-1, window_size)
    return windows, pad, original_length

def merge_windows(windows, original_length):
    reconstructed = np.asarray(windows).reshape(-1)
    return reconstructed[:original_length]

def select_sample(metadata, message_id, wpm, snr_db):
    rows = metadata[(metadata["message_id"] == message_id) & (metadata["wpm"] == wpm) & ( metadata["snr_requested_db"] == snr_db)]
    if len(rows) == 0:
        raise ValueError("No matching CW sample found.")

    if len(rows) > 1:
        raise ValueError(f"Expected one sample, found {len(rows)}.")

    return rows.iloc[0]

def topk_prune(coeffs, retention):
    coeffs = np.asarray(coeffs)
    if not 0 < retention <= 1:
        raise ValueError("retention must satisfy 0 < retention <= 1")

    n = len(coeffs)
    k = max(1, int(np.ceil(n * retention)),)
    if k >= n:
        return coeffs.copy(), np.arange(n)

    indices = np.argpartition(np.abs(coeffs), -k,)[-k:]
    compressed = np.zeros_like(coeffs)
    compressed[indices] = coeffs[indices]
    return compressed, np.sort(indices)

def topk_prune_fourier(coeffs, retention):
    coeffs = np.asarray(coeffs)
    if not 0 < retention <= 1:
        raise ValueError("retention must satisfy 0 < retention <= 1")

    n = len(coeffs)
    target = max(1, int(np.ceil(n * retention)),)
    if target >= n:
        return coeffs.copy(), np.arange(n)

    groups = []
    # DC
    groups.append(([0], np.abs(coeffs[0]) ** 2))
    # Nyquist
    if n % 2 == 0:
        k = n // 2
        groups.append(([k], np.abs(coeffs[k]) ** 2))

    # Conjugate frequency pairs
    for k in range(1, (n + 1) // 2):
        pair = [k, n - k]
        energy = (np.abs(coeffs[k]) ** 2 + np.abs(coeffs[n - k]) ** 2)
        groups.append((pair, energy))

    groups.sort(key=lambda item: item[1], reverse=True,)
    keep = []
    for indices, _ in groups:
        keep.extend(indices)
        if len(keep) >= target:
            break

    keep = np.array(sorted(set(keep)), dtype=int,)
    compressed = np.zeros_like(coeffs)
    compressed[keep] = coeffs[keep]
    return compressed, keep

def topk_prune_batch(coeffs, retention):
    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 2: raise ValueError("coeffs must be 2-D")
    n = coeffs.shape[1]
    k = max(1, int(np.ceil(n * retention)))
    if k >= n: return coeffs.copy(), np.full(coeffs.shape[0], n)

    indices = np.argpartition(np.abs(coeffs), -k, axis=1)[:, -k:]
    compressed = np.zeros_like(coeffs)
    rows = np.arange(coeffs.shape[0])[:, None]
    compressed[rows, indices] = coeffs[rows, indices]
    return compressed, np.full(coeffs.shape[0], k)

def topk_prune_fourier_batch(coeffs, retention):
    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 2: raise ValueError("coeffs must be 2-D")

    b, n = coeffs.shape
    if n % 2: raise ValueError("Expected even transform size")

    target = max(1, int(np.ceil(n * retention)))
    if target >= n: return coeffs.copy(), np.full(b, n)

    half = n // 2
    energies = np.empty((b, half + 1))
    energies[:, 0] = np.abs(coeffs[:, 0]) ** 2
    energies[:, half] = np.abs(coeffs[:, half]) ** 2
    energies[:, 1:half] = np.abs(coeffs[:, 1:half]) ** 2 + np.abs(coeffs[:, -1:half:-1]) ** 2

    sizes = np.full(half + 1, 2)
    sizes[[0, half]] = 1
    order = np.argsort(energies, axis=1)[:, ::-1]
    ordered_sizes = sizes[order]
    cumulative = np.cumsum(ordered_sizes, axis=1)
    selected_order = cumulative - ordered_sizes < target

    selected = np.zeros_like(energies, dtype=bool)
    rows = np.arange(b)[:, None]
    selected[rows, order] = selected_order

    keep = np.zeros((b, n), dtype=bool)
    keep[:, 0] = selected[:, 0]
    keep[:, half] = selected[:, half]
    keep[:, 1:half] = selected[:, 1:half]
    keep[:, -1:half:-1] = selected[:, 1:half]

    return np.where(keep, coeffs, 0), keep.sum(axis=1)

def prune_coeffs_batch(coeffs, retention, fourier=False):
    return topk_prune_fourier_batch(coeffs, retention) if fourier else topk_prune_batch(coeffs, retention)

def unpack_transform_result(result):
    # Quantum:
    # (coeffs, circuit, leakage)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], 0.0

    # Classical:
    # ndarray only
    return result, None, 0.0

def compress_window(window, forward_fn, inverse_fn, retention, fourier=False):
    coeffs, forward_qc, forward_leakage = unpack_transform_result(forward_fn(window))
    if fourier:
        compressed, kept = (topk_prune_fourier(coeffs, retention))
    else:
        compressed, kept = topk_prune(coeffs, retention)

    reconstructed, inverse_qc, inverse_leakage = unpack_transform_result((inverse_fn(compressed)))
    reconstructed = np.real_if_close(reconstructed, tol=1000).real
    return {
        "coeffs": coeffs,
        "compressed_coeffs": compressed,
        "reconstructed": reconstructed,
        "kept_indices": kept,
        "num_kept": len(kept),
        "actual_retention": (len(kept) / len(coeffs)),
        "forward_qc": forward_qc,
        "inverse_qc": inverse_qc,
        "forward_leakage": forward_leakage,
        "inverse_leakage": inverse_leakage,
    }

def waveform_metrics(original, reconstructed):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    error = (original - reconstructed)
    mae = np.mean(np.abs(error))
    mse = np.mean(np.abs(error) ** 2)
    rmse = np.sqrt(mse)
    signal_power = np.mean(np.abs(original) ** 2)
    noise_power = np.mean(np.abs(error) ** 2)

    if noise_power <= 1e-30:
        snr_db = np.inf
    elif signal_power <= 1e-30:
        snr_db = np.nan
    else:
        snr_db = 10 * np.log10(signal_power / noise_power)

    if (np.std(original) <= 1e-15 or np.std(reconstructed) <= 1e-15):
        correlation = np.nan
    else:
        correlation = np.corrcoef(original, reconstructed,)[0, 1]

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "snr_db": snr_db,
        "correlation": correlation,
    }

def prune_coeffs(coeffs, retention, fourier=False):
    return topk_prune_fourier(coeffs, retention) if fourier else topk_prune(coeffs, retention)

# def stabilize_amplitudes(x, zero_tol=1e-13):
#     a = np.asarray(x, dtype=np.complex128).copy()
#     if not np.all(np.isfinite(a)): 
#         raise ValueError("Non-finite amplitudes")
#     scale = np.max(np.abs(a))
#     if scale <= 1e-15: 
#         return None
#     a[np.abs(a) < zero_tol * scale] = 0.0
#     a = np.asarray(np.real_if_close(a, tol=1000), dtype=np.complex128)
#     norm = np.linalg.norm(a)
#     if norm <= 1e-15: return None
#     a /= norm
#     a /= np.sqrt(np.vdot(a, a).real)
#     return a

def build_transform_pipeline(x,transform):
    x=validate_window(x)
    x=np.asarray(x,dtype=np.complex128)
    norm=np.linalg.norm(x)
    n=int(np.log2(len(x)))

    if norm<=1e-15:
        return None,0.0,n

    amplitudes=stabilize_amplitudes(x)
    if amplitudes is None:
        return None,0.0,n

    prep_definition,perturbation,prep_method=synthesize_stateprep(amplitudes)

    if prep_method!="exact":
        print(f"Noise StatePrep fallback method={prep_method} perturbation={perturbation:.3e}")

    qc=QuantumCircuit(transform.num_qubits)
    qc.compose(prep_definition,qubits=range(n),inplace=True)
    qc.compose(transform,qubits=range(transform.num_qubits),inplace=True)

    return qc,norm,n

def compress_recording(signal, forward_fn, inverse_fn, retention_ratios, fourier=False):
    windows, pad, original_length = split_windows(signal)
    forward_cache = [unpack_transform_result(forward_fn(w)) for w in windows]
    results = {}

    for retention in retention_ratios:
        reconstructed_windows, kept_counts, inverse_leakages = [], [], []

        for coeffs, _, _ in forward_cache:
            compressed, kept = prune_coeffs(coeffs, retention, fourier)
            reconstructed, _, inverse_leakage = unpack_transform_result(inverse_fn(compressed))
            reconstructed_windows.append(np.real_if_close(reconstructed, tol=1000).real)
            kept_counts.append(len(kept))
            inverse_leakages.append(inverse_leakage)

        reconstructed = merge_windows(np.asarray(reconstructed_windows), original_length)
        forward_leakages = [leak for _, _, leak in forward_cache]

        results[retention] = {
            "reconstructed": reconstructed,
            "avg_kept": float(np.mean(kept_counts)),
            "actual_retention": float(np.mean(kept_counts) / windows.shape[1]),
            "forward_leakage_mean": float(np.mean(forward_leakages)),
            "forward_leakage_max": float(np.max(forward_leakages)),
            "inverse_leakage_mean": float(np.mean(inverse_leakages)),
            "inverse_leakage_max": float(np.max(inverse_leakages)),
        }

    return results


# Classical batch transforms
def classical_fft_batch(x): return fft(x, axis=1, norm="ortho", workers=SCIPY_WORKERS)
def classical_ifft_batch(x): return ifft(x, axis=1, norm="ortho", workers=SCIPY_WORKERS)
def classical_dct_batch(x): return dct(x, type=2, axis=1, norm="ortho", workers=SCIPY_WORKERS)
def classical_idct_batch(x): return idct(x, type=2, axis=1, norm="ortho", workers=SCIPY_WORKERS)
def classical_dst_batch(x): return dst(x, type=2, axis=1, norm="ortho", workers=SCIPY_WORKERS)
def classical_idst_batch(x): return idst(x, type=2, axis=1, norm="ortho", workers=SCIPY_WORKERS)


# Transform definitions from the notebook
def classical_fft(x):
    x = validate_window(x)
    return fft(x, norm="ortho")

def classical_ifft(coeffs):
    coeffs = validate_window(coeffs)
    return ifft(coeffs, norm="ortho")

def build_qft(num_qubits):
    qc = QuantumCircuit(num_qubits, name="QFT")
    for target in reversed(range(num_qubits)):
        qc.h(target)
        for control in reversed(range(target)):
            distance = target - control
            angle = np.pi / (2 ** distance)
            qc.cp(angle, control, target)

    for q in range(num_qubits // 2):
        qc.swap(q, num_qubits - q - 1)

    return qc

def build_iqft(num_qubits):
    qc = build_qft(num_qubits).inverse()
    qc.name = "IQFT"
    return qc

@lru_cache(maxsize=None)
def cached_qft(n):
    return build_qft(n)

@lru_cache(maxsize=None)
def cached_iqft(n):
    return build_iqft(n)

def quantum_fft(x):
    x = validate_window(x)
    num_qubits = int(np.log2(len(x)))
    transform = cached_iqft(num_qubits)
    result, qc = run_transform(x, transform)
    return result, qc, 0.0

def quantum_ifft(coeffs):
    coeffs = validate_window(coeffs)
    num_qubits = int(np.log2(len(coeffs)))
    transform = cached_qft(num_qubits)
    result, qc = run_transform(coeffs, transform)
    return result, qc, 0.0

def classical_haar(x):
    x = validate_window(x).astype(np.float64)
    approx = x.copy()
    details = []
    while len(approx) > 1:
        even = approx[0::2]
        odd = approx[1::2]
        next_approx = (even + odd) / np.sqrt(2.0)
        detail = (even - odd) / np.sqrt(2.0)
        details.append(detail)
        approx = next_approx

    return np.concatenate([approx, *details[::-1]])

def classical_ihaar(coeffs):
    coeffs = validate_window(coeffs).astype(np.float64)
    approx = coeffs[:1]
    offset = 1
    while offset < len(coeffs):
        n = len(approx)
        detail = coeffs[offset:offset + n]
        reconstructed = np.empty(2 * n, dtype=np.float64)
        reconstructed[0::2] = (approx + detail) / np.sqrt(2.0)
        reconstructed[1::2] = (approx - detail) / np.sqrt(2.0)
        approx = reconstructed
        offset += n

    return approx

def classical_haar_batch(x):
    approx = np.asarray(x, dtype=np.float64).copy()
    details = []

    while approx.shape[1] > 1:
        even, odd = approx[:, 0::2], approx[:, 1::2]
        details.append((even - odd) / np.sqrt(2.0))
        approx = (even + odd) / np.sqrt(2.0)

    return np.concatenate([approx, *details[::-1]], axis=1)

def classical_ihaar_batch(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.float64)
    approx, offset = coeffs[:, :1], 1

    while offset < coeffs.shape[1]:
        n = approx.shape[1]
        detail = coeffs[:, offset:offset + n]
        reconstructed = np.empty((coeffs.shape[0], 2 * n))
        reconstructed[:, 0::2] = (approx + detail) / np.sqrt(2.0)
        reconstructed[:, 1::2] = (approx - detail) / np.sqrt(2.0)
        approx, offset = reconstructed, offset + n

    return approx

def build_qhaar(num_qubits):
    qc = QuantumCircuit(num_qubits, name="QDWT-Haar")

    for level in range(num_qubits):
        active_qubits = (num_qubits - level)

        controls = list(range(active_qubits, num_qubits))

        # Haar average/detail pair
        append_zero_controlled_h(qc, controls, target=0)

        # Pack the new detail bit at the
        # upper edge of the active region.
        # q0 -> q(active_qubits - 1)
        if active_qubits > 1:
            for q in range(active_qubits - 1):
                append_zero_controlled_swap(qc, controls, q, q + 1,)
    return qc

def build_iqhaar(num_qubits):
    qc = build_qhaar(num_qubits).inverse()
    qc.name = "IQDWT-Haar"
    return qc

@lru_cache(maxsize=None)
def cached_qhaar(n):
    return build_qhaar(n)

@lru_cache(maxsize=None)
def cached_iqhaar(n):
    return build_iqhaar(n)

def quantum_haar(x):
    x = validate_window(x)
    n = int(np.log2(len(x)))
    result, qc = run_transform(x, cached_qhaar(n))
    return result, qc, 0.0

def quantum_ihaar(coeffs):
    coeffs = validate_window(coeffs)
    n = int(np.log2(len(coeffs)))
    result, qc = run_transform(coeffs, cached_iqhaar(n))
    return result, qc, 0.0

def classical_dct(x):
    x = validate_window(x)
    return dct(x, type=2, norm="ortho")

def classical_idct(coeffs):
    coeffs = validate_window(coeffs)
    return idct(coeffs, type=2, norm="ortho")

def build_qdct(num_data_qubits):
    num_total_qubits = (num_data_qubits + 1)
    qc = QuantumCircuit(num_total_qubits, name="QDCT-II")
    data = list(range(num_data_qubits))
    ctrl = num_data_qubits
    append_qct_qst_core(qc, data, ctrl)
    return qc

def build_iqdct(num_data_qubits):
    qc = build_qdct(num_data_qubits).inverse()
    qc.name = "IQDCT"
    return qc

@lru_cache(maxsize=None)
def cached_qdct(n):
    return build_qdct(n)

@lru_cache(maxsize=None)
def cached_iqdct(n):
    return build_iqdct(n)

def quantum_dct(x):
    x = validate_window(x)
    n = int(np.log2(len(x)))
    return run_branch_transform(x, cached_qdct(int(n)))

def quantum_idct(coeffs):
    coeffs = validate_window(coeffs)
    n = int(np.log2(len(coeffs)))
    return run_branch_transform(coeffs, cached_iqdct(int(n)))

def classical_dst(x):
    x = validate_window(x)
    return dst(x, type=2, norm="ortho")

def classical_idst(coeffs):
    coeffs = validate_window(coeffs)
    return idst(coeffs, type=2, norm="ortho")

def build_qdst(num_data_qubits):
    num_total_qubits = (num_data_qubits + 1)
    qc = QuantumCircuit(num_total_qubits, name="QDST-II")
    data = list(range(num_data_qubits))
    ctrl = num_data_qubits
    # Select sine branch
    qc.x(ctrl)
    # Shared:
    # U_N^\dagger QFT_{2N} V_N
    append_qct_qst_core(qc, data, ctrl)
    # Remove sine-branch phase
    qc.z(ctrl)
    # Restore branch qubit to |0>
    qc.x(ctrl)
    return qc

def build_iqdst(num_data_qubits):
    qc = build_qdst(num_data_qubits).inverse()
    qc.name = "IQDST"
    return qc

@lru_cache(maxsize=None)
def cached_qdst(n):
    return build_qdst(n)

@lru_cache(maxsize=None)
def cached_iqdst(n):
    return build_iqdst(n)

def quantum_dst(x):
    x = validate_window(x)
    n = int(np.log2(len(x)))
    return run_branch_transform(x, cached_qdst(int(n)))

def quantum_idst(coeffs):
    coeffs = validate_window(coeffs)
    n = int(np.log2(len(coeffs)))
    return run_branch_transform(coeffs, cached_iqdst(int(n)))


CLASSICAL_BATCH_TRANSFORMS = {
    "FFT": (classical_fft_batch, classical_ifft_batch, True),
    "DCT": (classical_dct_batch, classical_idct_batch, False),
    "DST": (classical_dst_batch, classical_idst_batch, False),
    "DWT": (classical_haar_batch, classical_ihaar_batch, False),
}

QUANTUM_BATCH_TRANSFORMS = {
    "QFT":  (cached_iqft,  cached_qft,  True,  False),
    "QDCT": (cached_qdct,  cached_iqdct, False, True),
    "QDST": (cached_qdst,  cached_iqdst, False, True),
    "QDWT": (cached_qhaar, cached_iqhaar, False, False),
}

BATCH_TRANSFORMS = [*CLASSICAL_BATCH_TRANSFORMS, *QUANTUM_BATCH_TRANSFORMS]


def run_quantum_batch(vectors, transform_builder, branch=False):
    vectors = np.asarray(vectors)
    N = vectors.shape[1]
    n = int(np.log2(N))
    transform = transform_builder(n)

    outputs = np.zeros((len(vectors), N), dtype=np.complex128)
    leakages = np.zeros(len(vectors), dtype=np.float64)

    items = [(i, vectors[i]) for i in range(len(vectors))]
    t = perf_counter()
    if STATEPREP_EXECUTOR is None or QISKIT_STATEPREP_PROCS <= 1:
        built = [build_stateprep_item(item) for item in items]
    else:
        built = list(STATEPREP_EXECUTOR.map(build_stateprep_item, items, chunksize=1))
    print(f"    stateprep={perf_counter() - t:.2f}s")

    for start in range(0, len(built), Q_BATCH_SIZE):
        stop = min(start + Q_BATCH_SIZE, len(built))
        circuits, metadata_batch = [], []

        for i,norm,prep_definition,perturbation,retry_decimals in built[start:stop]:
            if prep_definition is None:
                continue

            if retry_decimals != "exact":
                print(f"StatePrep retry window={i} method={retry_decimals} perturbation={perturbation:.3e}")

            qc=QuantumCircuit(transform.num_qubits)
            qc.compose(prep_definition,qubits=range(n),inplace=True)
            qc.compose(transform,qubits=range(transform.num_qubits),inplace=True)
            qc.save_statevector()

            circuits.append(qc)
            metadata_batch.append((i,norm))

        if not circuits:
            continue

        t = perf_counter()
        tqc = transpile(circuits, CPU_QSIM, optimization_level=0, num_processes=QISKIT_TRANSPILE_PROCS)
        transpile_time = perf_counter() - t

        t = perf_counter()
        result = CPU_QSIM.run(tqc).result()
        simulation_time = perf_counter() - t
        print(f"    batch {start // Q_BATCH_SIZE + 1:02d} transpile={transpile_time:.2f}s aer={simulation_time:.2f}s")

        for j, (i, norm) in enumerate(metadata_batch):
            state = np.asarray(result.data(j)["statevector"])
            if branch:
                output_branch = state[:N]
                outputs[i] = output_branch * norm
                leakages[i] = max(0.0, 1.0 - np.sum(np.abs(output_branch) ** 2))
            else:
                outputs[i] = state * norm

    return outputs, leakages


def compress_recording_classical(signal, name, retention_ratios):
    windows, _, original_length = split_windows(signal)
    forward_fn, inverse_fn, fourier = CLASSICAL_BATCH_TRANSFORMS[name]
    coeffs = forward_fn(windows)
    results = {}

    for retention in retention_ratios:
        compressed, kept_counts = prune_coeffs_batch(coeffs, retention, fourier)
        reconstructed_windows = np.real_if_close(inverse_fn(compressed), tol=1000).real
        reconstructed = merge_windows(reconstructed_windows, original_length)
        results[retention] = {
            "reconstructed": reconstructed,
            "avg_kept": float(np.mean(kept_counts)),
            "actual_retention": float(np.mean(kept_counts) / windows.shape[1]),
            "forward_leakage_mean": 0.0,
            "forward_leakage_max": 0.0,
            "inverse_leakage_mean": 0.0,
            "inverse_leakage_max": 0.0,
        }

    return results

def estimated_qpu_duration_us(qc,target):
    try:
        return float(qc.estimate_duration(target,unit="u"))
    except Exception as e:
        print(f"QPU duration unavailable: {e}")
        return np.nan


def compress_recording_quantum(signal, name, retention_ratios):
    windows, _, original_length = split_windows(signal)
    forward_builder, inverse_builder, fourier, branch = QUANTUM_BATCH_TRANSFORMS[name]
    print(f"  {name}: {len(windows)} windows")

    t = perf_counter()
    coeffs, forward_leakages = run_quantum_batch(windows, forward_builder, branch)
    print(f"  forward: {perf_counter() - t:.1f}s")
    results = {}

    for j, retention in enumerate(retention_ratios, 1):
        print(f"  inverse {j}/{len(retention_ratios)} retention={retention:.0%}", flush=True)
        compressed, kept_counts = prune_coeffs_batch(coeffs, retention, fourier)

        t = perf_counter()
        reconstructed_windows, inverse_leakages = run_quantum_batch(compressed, inverse_builder, branch)
        elapsed = perf_counter() - t
        reconstructed = merge_windows(np.real_if_close(reconstructed_windows, tol=1000).real, original_length)

        results[retention] = {
            "reconstructed": reconstructed,
            "avg_kept": float(np.mean(kept_counts)),
            "actual_retention": float(np.mean(kept_counts) / windows.shape[1]),
            "forward_leakage_mean": float(np.mean(forward_leakages)),
            "forward_leakage_max": float(np.max(forward_leakages)),
            "inverse_leakage_mean": float(np.mean(inverse_leakages)),
            "inverse_leakage_max": float(np.max(inverse_leakages)),
        }
        print(f"  -> {elapsed:.1f}s")

    return results


def run_cw_experiment(clean_signal, noisy_signal, row, transform_names, include_clean=True):
    rows, reconstructed_signals = [], {}
    signals = {"clean": clean_signal, "noisy": noisy_signal} if include_clean else {"noisy": noisy_signal}

    for signal_type, signal in signals.items():
        for name in transform_names:
            print(f"msg={row['message_id']} wpm={row['wpm']} snr={row['snr_requested_db']} | {signal_type} | {name}")

            if name in CLASSICAL_BATCH_TRANSFORMS:
                implementation = "classical"
                results = compress_recording_classical(signal, name, RETENTION_RATIOS)
            elif name in QUANTUM_BATCH_TRANSFORMS:
                implementation = "quantum"
                results = compress_recording_quantum(signal, name, RETENTION_RATIOS)
            else:
                raise KeyError(f"Unknown transform: {name}")

            for retention, result in results.items():
                input_m = waveform_metrics(signal, result["reconstructed"])
                clean_m = waveform_metrics(clean_signal, result["reconstructed"])
                rows.append({
                    "message_id": row["message_id"],
                    "wpm": row["wpm"],
                    "snr_requested_db": row["snr_requested_db"],
                    "signal_type": signal_type,
                    "transform": name,
                    "implementation": implementation,
                    "retention_requested": retention,
                    "retention_actual": result["actual_retention"],
                    "avg_kept": result["avg_kept"],
                    "rmse": input_m["rmse"],
                    "snr_db": input_m["snr_db"],
                    "correlation": input_m["correlation"],
                    "rmse_vs_clean": clean_m["rmse"],
                    "snr_vs_clean_db": clean_m["snr_db"],
                    "corr_vs_clean": clean_m["correlation"],
                })
                reconstructed_signals[(signal_type, name, retention)] = result["reconstructed"]

    return pd.DataFrame(rows), reconstructed_signals

def save_noise_plots(noise_results, plots_dir):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = noise_results.copy()

    def grouped_bar(subdf, value_col, title, ylabel, filename, ylim=None):
        devices = list(subdf["device"].drop_duplicates())
        transforms = list(subdf["transform"].drop_duplicates())
        x = np.arange(len(transforms))
        width = 0.8 / max(1, len(devices))

        plt.figure(figsize=(10, 6))
        for i, device in enumerate(devices):
            vals = []
            for t in transforms:
                row=subdf[
                    (subdf["device"]==device)
                    & (subdf["transform"]==t)
                ]

                vals.append(
                    row[value_col].mean()
                    if not row.empty
                    else np.nan
                )
            plt.bar(x + (i - (len(devices)-1)/2)*width, vals, width=width, label=device)

        plt.xticks(x, transforms)
        plt.xlabel("Quantum transform")
        plt.ylabel(ylabel)
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=350, bbox_inches="tight")
        plt.close()

    # fidelity / leakage for clean and noisy inputs separately
    for signal_type in ["clean", "noisy"]:
        sub = df[df["signal_type"] == signal_type].copy()
        if sub.empty:
            continue

        grouped_bar(sub, "fidelity_mean", f"Mean Fidelity ({signal_type})", "Mean fidelity", f"14_noise_fidelity_{signal_type}.png", ylim=(0, 1.01))
        grouped_bar(sub, "leakage_mean", f"Mean Leakage ({signal_type})", "Mean leakage", f"15_noise_leakage_{signal_type}.png")
        grouped_bar(sub, "depth_mean", f"Circuit Depth ({signal_type})", "Mean depth", f"16_noise_depth_{signal_type}.png")
        grouped_bar(sub, "size_mean", f"Circuit Size ({signal_type})", "Mean gate count", f"17_noise_size_{signal_type}.png")
        grouped_bar(sub, "twoq_mean", f"Two-Qubit Gate Count ({signal_type})", "Mean 2-qubit gate count", f"18_noise_twoqubit_{signal_type}.png")

    # one extra plot: fidelity drop from ideal to hardware, clean/noisy separately
    for signal_type in ["clean", "noisy"]:
        sub = df[df["signal_type"] == signal_type].copy()
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="transform", columns="device", values="fidelity_mean", aggfunc="mean")
        if {"ideal", "FakeSherbrooke"}.issubset(set(pivot.columns)):
            delta = (pivot["ideal"] - pivot["FakeSherbrooke"]).reindex(pivot.index)

            plt.figure(figsize=(10, 6))
            plt.bar(delta.index, delta.values)
            plt.xlabel("Quantum transform")
            plt.ylabel("Fidelity drop")
            plt.title(f"Fidelity Drop: Ideal vs FakeSherbrooke ({signal_type})")
            plt.tight_layout()
            plt.savefig(plots_dir / f"19_noise_fidelity_drop_{signal_type}.png", dpi=350, bbox_inches="tight")
            plt.close()

    fake=df[df["device"]=="FakeSherbrooke"].copy()

    if not fake.empty:
        # 35 — Two-qubit cost vs fidelity
        plt.figure(figsize=(9,6))
        sns.scatterplot(
            data=fake,
            x="twoq_mean",
            y="fidelity_mean",
            hue="transform",
            style="signal_type",
            s=120,
        )
        plt.xlabel("Mean two-qubit gate count")
        plt.ylabel("Mean state fidelity")
        plt.title("Quantum Resource Cost vs Noise Fidelity")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"35_resource_cost_vs_fidelity.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

        # 36 — Depth vs fidelity
        plt.figure(figsize=(9,6))
        sns.scatterplot(
            data=fake,
            x="depth_mean",
            y="fidelity_mean",
            hue="transform",
            style="signal_type",
            s=120,
        )
        plt.xlabel("Mean transpiled circuit depth")
        plt.ylabel("Mean state fidelity")
        plt.title("Circuit Depth vs FakeSherbrooke Fidelity")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"36_depth_vs_fidelity.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

def save_plots(full_results,plots_dir):
    plots_dir=Path(plots_dir)
    plots_dir.mkdir(parents=True,exist_ok=True)

    df=full_results.copy().replace([np.inf,-np.inf],np.nan)
    df["retention_pct"]=df["retention_requested"]*100
    df["actual_retention_pct"]=df["retention_actual"]*100

    family_map={
        "FFT":"Fourier","QFT":"Fourier",
        "DCT":"Cosine","QDCT":"Cosine",
        "DST":"Sine","QDST":"Sine",
        "DWT":"Haar","QDWT":"Haar",
    }
    df["family"]=df["transform"].map(family_map)

    noisy_df=df[df["signal_type"]=="noisy"].copy()
    clean=df[df["signal_type"]=="clean"].copy()
    family_order=["Fourier","Cosine","Sine","Haar"]

    sns.set_theme(style="whitegrid",context="paper")

    def save_grid(g,name):
        g.figure.tight_layout()
        g.figure.savefig(plots_dir/name,dpi=350,bbox_inches="tight")
        plt.close(g.figure)

    def save_current(name):
        plt.tight_layout()
        plt.savefig(plots_dir/name,dpi=350,bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------------------------
    # Classical vs Quantum paired comparison plots
    # -------------------------------------------------------------------------

    if not clean.empty:
        g=sns.relplot(data=clean,x="retention_pct",y="rmse",hue="implementation",col="family",col_order=family_order,kind="line",marker="o",estimator="mean",errorbar="sd",height=4,aspect=1.05)
        g.set_axis_labels("Retained coefficients (%)","RMSE")
        g.set_titles("{col_name}: Classical vs Quantum")
        for ax in g.axes.flat: ax.invert_xaxis()
        save_grid(g,"20_classical_quantum_clean_rmse.png")

    if not noisy_df.empty:
        g=sns.relplot(data=noisy_df,x="retention_pct",y="rmse_vs_clean",hue="implementation",col="family",col_order=family_order,kind="line",marker="o",estimator="mean",errorbar="sd",height=4,aspect=1.05)
        g.set_axis_labels("Retained coefficients (%)","RMSE vs clean CW")
        g.set_titles("{col_name}: Classical vs Quantum")
        for ax in g.axes.flat: ax.invert_xaxis()
        save_grid(g,"21_classical_quantum_noisy_rmse.png")

        g=sns.relplot(data=noisy_df,x="retention_pct",y="snr_vs_clean_db",hue="implementation",col="family",col_order=family_order,kind="line",marker="o",estimator="mean",errorbar="sd",height=4,aspect=1.05)
        g.set_axis_labels("Retained coefficients (%)","SNR vs clean CW (dB)")
        g.set_titles("{col_name}: Classical vs Quantum")
        for ax in g.axes.flat: ax.invert_xaxis()
        save_grid(g,"22_classical_quantum_noisy_snr.png")

        g=sns.relplot(data=noisy_df,x="retention_pct",y="corr_vs_clean",hue="implementation",col="family",col_order=family_order,kind="line",marker="o",estimator="mean",errorbar="sd",height=4,aspect=1.05)
        g.set_axis_labels("Retained coefficients (%)","Correlation vs clean CW")
        g.set_titles("{col_name}: Classical vs Quantum")
        for ax in g.axes.flat:
            ax.invert_xaxis()
            ax.set_ylim(0,1.01)
        save_grid(g,"23_classical_quantum_noisy_correlation.png")

        pair_keys=["message_id","wpm","snr_requested_db","retention_requested","retention_pct","family"]

        rmse_pair=noisy_df.pivot_table(index=pair_keys,columns="implementation",values="rmse_vs_clean",aggfunc="mean").reset_index()
        if {"classical","quantum"}.issubset(rmse_pair.columns):
            rmse_pair["delta_rmse"]=rmse_pair["quantum"]-rmse_pair["classical"]
            g=sns.relplot(data=rmse_pair,x="retention_pct",y="delta_rmse",hue="family",hue_order=family_order,kind="line",marker="o",estimator="mean",errorbar="sd",height=5,aspect=1.4)
            g.set_axis_labels("Retained coefficients (%)","ΔRMSE (Quantum − Classical)")
            g.ax.axhline(0,linestyle="--",linewidth=1)
            g.ax.invert_xaxis()
            save_grid(g,"24_quantum_classical_delta_rmse.png")

        snr_pair=noisy_df.pivot_table(index=pair_keys,columns="implementation",values="snr_vs_clean_db",aggfunc="mean").reset_index()
        if {"classical","quantum"}.issubset(snr_pair.columns):
            snr_pair["delta_snr_db"]=snr_pair["quantum"]-snr_pair["classical"]
            g=sns.relplot(data=snr_pair,x="retention_pct",y="delta_snr_db",hue="family",hue_order=family_order,kind="line",marker="o",estimator="mean",errorbar="sd",height=5,aspect=1.4)
            g.set_axis_labels("Retained coefficients (%)","ΔSNR (Quantum − Classical) [dB]")
            g.ax.axhline(0,linestyle="--",linewidth=1)
            g.ax.invert_xaxis()
            save_grid(g,"25_quantum_classical_delta_snr.png")

    # -------------------------------------------------------------------------
    # General transform plots
    # -------------------------------------------------------------------------

    if not df.empty:
        g=sns.relplot(data=df,x="retention_pct",y="rmse",hue="transform",col="signal_type",kind="line",marker="o",height=4.5,aspect=1.25)
        g.set_axis_labels("Retained coefficients (%)","RMSE")
        g.set_titles("{col_name} CW")
        for ax in g.axes.flat: ax.invert_xaxis()
        save_grid(g,"01_rmse_vs_retention.png")

        g=sns.relplot(data=df,x="retention_pct",y="snr_db",hue="transform",col="signal_type",kind="line",marker="o",height=4.5,aspect=1.25)
        g.set_axis_labels("Retained coefficients (%)","Reconstruction SNR (dB)")
        g.set_titles("{col_name} CW")
        for ax in g.axes.flat: ax.invert_xaxis()
        save_grid(g,"02_snr_vs_retention.png")

        g=sns.relplot(data=df,x="retention_pct",y="correlation",hue="transform",col="signal_type",kind="line",marker="o",height=4.5,aspect=1.25)
        g.set_axis_labels("Retained coefficients (%)","Correlation")
        g.set_titles("{col_name} CW")
        for ax in g.axes.flat:
            ax.invert_xaxis()
            ax.set_ylim(0,1.01)
        save_grid(g,"04_correlation_vs_retention.png")

    # -------------------------------------------------------------------------
    # Noisy CW plots
    # -------------------------------------------------------------------------

    if not noisy_df.empty:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=noisy_df,x="retention_pct",y="snr_vs_clean_db",hue="transform",marker="o")
        plt.xlabel("Retained coefficients (%)")
        plt.ylabel("SNR vs clean CW (dB)")
        plt.title("Noisy CW Reconstruction Quality vs Clean Reference")
        plt.gca().invert_xaxis()
        save_current("03_noisy_snr_vs_clean.png")

        plt.figure(figsize=(10,6))
        sns.lineplot(data=noisy_df,x="retention_pct",y="rmse_vs_clean",hue="family",style="implementation",markers=True)
        plt.xlabel("Retained coefficients (%)")
        plt.ylabel("RMSE vs clean CW")
        plt.title("Classical vs Quantum Compression of Noisy CW")
        plt.gca().invert_xaxis()
        save_current("06_family_noisy_rmse.png")

        noisy10=noisy_df[noisy_df["snr_requested_db"]==10]
        if not noisy10.empty:
            n_wpm=noisy10["wpm"].nunique()
            g=sns.relplot(
                data=noisy10,
                x="retention_pct",
                y="snr_vs_clean_db",
                hue="transform",
                col="wpm",
                col_wrap=min(3,n_wpm),
                kind="line",
                marker="o",
                estimator="mean",
                errorbar="sd",
                height=3.8,
                aspect=1.15,
            )
            g.set_axis_labels(
                "Retained coefficients (%)",
                "SNR vs clean CW (dB)",
            )
            g.set_titles("{col_name} WPM — 10 dB input")

            for ax in g.axes.flat:
                ax.invert_xaxis()

            save_grid(g,"08_noisy_snr_by_wpm_10db.png")

        five=noisy_df[np.isclose(noisy_df["retention_requested"],0.05)]
        if not five.empty:
            g=sns.relplot(data=five,x="wpm",y="snr_vs_clean_db",hue="transform",col="snr_requested_db",col_wrap=4,kind="line",marker="o",estimator="mean",errorbar="sd",height=3.4,aspect=1.05)
            g.set_axis_labels("WPM","Reconstructed SNR vs clean (dB)")
            g.set_titles("Input SNR = {col_name} dB")
            save_grid(g,"09_wpm_by_snr_at_5pct.png")

        heat=noisy_df.pivot_table(index="transform",columns="retention_pct",values="snr_vs_clean_db",aggfunc="mean")
        if not heat.empty:
            plt.figure(figsize=(11,5))
            sns.heatmap(heat,annot=True,fmt=".1f",cmap="viridis")
            plt.xlabel("Retained coefficients (%)")
            plt.ylabel("Transform")
            plt.title("Noisy CW Reconstruction SNR vs Clean Reference (dB)")
            save_current("11_heatmap_noisy_snr.png")

    # -------------------------------------------------------------------------
    # Clean CW plots
    # -------------------------------------------------------------------------

    if not clean.empty:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=clean,x="retention_pct",y="rmse",hue="family",style="implementation",markers=True)
        plt.xlabel("Retained coefficients (%)")
        plt.ylabel("RMSE")
        plt.title("Classical vs Quantum Transform Compression — Clean CW")
        plt.gca().invert_xaxis()
        save_current("05_family_clean_rmse.png")

        n_wpm=clean["wpm"].nunique()
        g=sns.relplot(
            data=clean,
            x="retention_pct",
            y="rmse_vs_clean",
            hue="transform",
            col="wpm",
            col_wrap=min(3,n_wpm),
            kind="line",
            marker="o",
            estimator="mean",
            errorbar=None,
            height=3.8,
            aspect=1.15,
        )
        g.set_axis_labels("Retained coefficients (%)","RMSE")
        g.set_titles("{col_name} WPM")
        for ax in g.axes.flat: ax.invert_xaxis()
        save_grid(g,"07_clean_rmse_by_wpm.png")

        heat=clean.pivot_table(index="transform",columns="retention_pct",values="rmse",aggfunc="mean")
        if not heat.empty:
            plt.figure(figsize=(11,5))
            sns.heatmap(heat,annot=True,fmt=".3f",cmap="viridis_r")
            plt.xlabel("Retained coefficients (%)")
            plt.ylabel("Transform")
            plt.title("CW Reconstruction RMSE")
            save_current("10_heatmap_clean_rmse.png")

        for transform in sorted(clean["transform"].dropna().unique()):
            subset=clean[clean["transform"]==transform]
            heat=subset.pivot_table(index="wpm",columns="retention_pct",values="rmse_vs_clean",aggfunc="mean")
            if heat.empty: continue
            plt.figure(figsize=(10,4))
            sns.heatmap(heat,annot=True,fmt=".3f",cmap="viridis_r")
            plt.title(f"{transform} — RMSE by WPM and Retention")
            plt.xlabel("Retained coefficients (%)")
            plt.ylabel("WPM")
            save_current(f"12_heatmap_{transform.lower()}_wpm_retention.png")

        ten=clean[np.isclose(clean["retention_requested"],0.10)]
        if not ten.empty:
            plt.figure(figsize=(9,5))
            sns.barplot(data=ten,x="transform",y="rmse")
            plt.xlabel("Transform")
            plt.ylabel("RMSE")
            plt.title("CW Reconstruction at 10% Coefficient Retention")
            save_current("13_bar_rmse_10pct.png")

    # -------------------------------------------------------------------------
    # Requested vs actual coefficient retention
    # -------------------------------------------------------------------------

    if not df.empty:
        plt.figure(figsize=(9,6))
        sns.lineplot(data=df,x="retention_pct",y="actual_retention_pct",hue="transform",marker="o")
        plt.plot([0,100],[0,100],"--",linewidth=1)
        plt.xlabel("Requested retention (%)")
        plt.ylabel("Actual retention (%)")
        plt.title("Requested vs Actual Coefficient Retention")
        save_current("26_requested_vs_actual_retention.png")

    print(f"Saved plots to: {plots_dir.resolve()}")

def circuit_resource_metrics(qc):
    ops=dict(qc.count_ops())

    oneq=sum(
        1 for inst in qc.data
        if inst.operation.num_qubits==1
    )
    twoq=sum(
        1 for inst in qc.data
        if inst.operation.num_qubits==2
    )
    multiq=sum(
        1 for inst in qc.data
        if inst.operation.num_qubits>2
    )

    try:
        twoq_depth=qc.depth(
            lambda inst: inst.operation.num_qubits==2
        )
    except Exception:
        twoq_depth=np.nan

    return {
        "qubits":qc.num_qubits,
        "depth":qc.depth(),
        "size":qc.size(),
        "oneq":oneq,
        "twoq":twoq,
        "multiq":multiq,
        "twoq_depth":twoq_depth,
        "ops":ops,
    }


def build_resource_pipeline(window,transform):
    x=validate_window(window)
    x=np.asarray(x,dtype=np.complex128)
    norm=np.linalg.norm(x)
    n=int(np.log2(len(x)))

    if norm<=1e-15:
        raise ValueError("Representative resource window is empty.")

    amplitudes=stabilize_amplitudes(x)

    if amplitudes is None:
        raise ValueError("Could not prepare representative amplitudes.")

    prep_definition,perturbation,prep_method=synthesize_stateprep(amplitudes)

    if prep_method!="exact":
        print(
            f"Resource StatePrep fallback "
            f"method={prep_method} "
            f"perturbation={perturbation:.3e}"
        )

    qc=QuantumCircuit(transform.num_qubits)
    qc.compose(
        prep_definition,
        qubits=range(n),
        inplace=True,
    )
    qc.compose(
        transform,
        qubits=range(transform.num_qubits),
        inplace=True,
    )

    return qc

def save_circuit_chunks(qc,path_prefix,chunk_size=400,max_chunks=3):
    path_prefix=Path(path_prefix)
    total=len(qc.data)
    chunks=min(max_chunks,int(np.ceil(total/chunk_size)))

    for ci in range(chunks):
        start=ci*chunk_size
        stop=min(start+chunk_size,total)

        sub=QuantumCircuit(
            qc.num_qubits,
            qc.num_clbits,
        )

        for inst in qc.data[start:stop]:
            qargs=[
                sub.qubits[
                    qc.find_bit(q).index
                ]
                for q in inst.qubits
            ]

            cargs=[
                sub.clbits[
                    qc.find_bit(c).index
                ]
                for c in inst.clbits
            ]

            sub.append(
                inst.operation,
                qargs,
                cargs,
            )

        fig=sub.draw(
            output="mpl",
            fold=40,
            idle_wires=False,
        )

        filename=(
            path_prefix.parent/
            f"{path_prefix.name}_part{ci+1:02d}.png"
        )

        try:
            fig.savefig(
                filename,
                dpi=250,
                bbox_inches="tight",
            )
        except RuntimeError as e:
            print(
                f"    circuit chunk {ci+1} "
                f"PNG skipped: {e}"
            )

        plt.close(fig)

        print(
            f"    saved circuit chunk "
            f"{ci+1}/{chunks}: "
            f"operations {start}-{stop-1}"
        )

def save_block_circuit_drawing(qc,path,label):
    wrapper=QuantumCircuit(qc.num_qubits)
    wrapper.append(qc.to_instruction(label=label), range(qc.num_qubits))
    fig=wrapper.draw(output="mpl", fold=-1, idle_wires=False)
    fig.savefig(path, dpi=350, bbox_inches="tight")
    plt.close(fig)

def save_expanded_circuit_drawing(qc,path,reps=2,max_ops=1200):
    path=Path(path)
    expanded=qc.decompose(reps=reps)

    print(
        f"    expanded drawing: reps={reps} "
        f"depth={expanded.depth()} "
        f"size={expanded.size()}"
    )

    if expanded.size()>max_ops:
        print(
            f"    expanded circuit too large to draw "
            f"({expanded.size()} ops > {max_ops}); "
            f"saving representative chunks instead"
        )
        save_circuit_chunks(
            expanded,
            path.parent/path.stem,
            chunk_size=400,
            max_chunks=3,
        )
        return

    fig=expanded.draw(
        output="mpl",
        fold=40,
        idle_wires=False,
    )

    try:
        fig.savefig(
            path.with_suffix(".png"),
            dpi=250,
            bbox_inches="tight",
        )
    except RuntimeError as e:
        print(f"    PNG expanded drawing skipped: {e}")

    try:
        fig.savefig(
            path.with_suffix(".pdf"),
            bbox_inches="tight",
        )
    except Exception as e:
        print(f"    PDF expanded drawing skipped: {e}")

    plt.close(fig)

def save_native_circuit_drawing(qc,path,basis_gates=None,backend=None,opt_level=1,max_ops=1200):
    path=Path(path)

    if backend is not None:
        native=transpile(
            qc,
            backend,
            optimization_level=opt_level,
            seed_transpiler=SEED,
        )
    else:
        native=transpile(
            qc,
            basis_gates=basis_gates,
            optimization_level=opt_level,
            seed_transpiler=SEED,
        )

    print(
        f"    native drawing: "
        f"depth={native.depth()} "
        f"size={native.size()} "
        f"ops={dict(native.count_ops())}"
    )

    if native.size()>max_ops:
        print(
            f"    native circuit too large to draw "
            f"({native.size()} ops > {max_ops}); "
            f"saving representative chunks"
        )

        save_circuit_chunks(
            native,
            path.parent/path.stem,
            chunk_size=400,
            max_chunks=3,
        )
        return

    fig=native.draw(
        output="mpl",
        fold=40,
        idle_wires=False,
    )

    try:
        fig.savefig(
            path.with_suffix(".png"),
            dpi=250,
            bbox_inches="tight",
        )
    except RuntimeError as e:
        print(f"    native PNG skipped: {e}")

    plt.close(fig)


def save_detailed_circuit_drawing(qc,path):
    fig=qc.draw(
        output="mpl",
        fold=40,
        idle_wires=False,
    )
    fig.savefig(
        path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

def save_quantum_circuit_resources(representative_window,output_dir):
    output_dir=Path(output_dir)
    plots_dir=output_dir/"plots"
    circuits_dir=plots_dir/"circuits"

    plots_dir.mkdir(parents=True,exist_ok=True)
    circuits_dir.mkdir(parents=True,exist_ok=True)

    n=int(np.log2(WINDOW_SIZE))
    fake=FakeSherbrooke()

    transform_builders={
        "QFT":build_iqft,
        "QDCT":build_qdct,
        "QDST":build_qdst,
        "QDWT":build_qhaar,
    }

    rows=[]
    gate_rows=[]

    for name,builder in transform_builders.items():
        print(f"Resource analysis: {name}")

        t0=perf_counter()
        transform=builder(n)
        transform_build_time=perf_counter()-t0

        t0=perf_counter()
        transform_basis=transpile(
            transform,
            basis_gates=RESOURCE_BASIS,
            optimization_level=RESOURCE_OPT_LEVEL,
            seed_transpiler=SEED,
        )
        transform_transpile_time=perf_counter()-t0

        t0=perf_counter()
        pipeline=build_resource_pipeline(
            representative_window,
            transform,
        )
        pipeline_build_time=perf_counter()-t0

        t0=perf_counter()
        pipeline_basis=transpile(
            pipeline,
            basis_gates=RESOURCE_BASIS,
            optimization_level=RESOURCE_OPT_LEVEL,
            seed_transpiler=SEED,
        )
        pipeline_transpile_time=perf_counter()-t0

        t0=perf_counter()
        device_pipeline=transpile(
            pipeline,
            fake,
            optimization_level=RESOURCE_OPT_LEVEL,
            seed_transpiler=SEED,
        )
        device_transpile_time=perf_counter()-t0

        qpu_duration_us=estimated_qpu_duration_us(
            device_pipeline,
            fake.target,
        )

        circuits={
            "transform_only":{
                "qc":transform_basis,
                "build_time_s":transform_build_time,
                "transpile_time_s":transform_transpile_time,
                "estimated_qpu_us":np.nan,
            },
            "stateprep_transform":{
                "qc":pipeline_basis,
                "build_time_s":pipeline_build_time,
                "transpile_time_s":pipeline_transpile_time,
                "estimated_qpu_us":np.nan,
            },
            "FakeSherbrooke":{
                "qc":device_pipeline,
                "build_time_s":pipeline_build_time,
                "transpile_time_s":device_transpile_time,
                "estimated_qpu_us":qpu_duration_us,
            },
        }

        for scope,item in circuits.items():
            qc=item["qc"]
            m=circuit_resource_metrics(qc)

            rows.append({
                "transform":name,
                "scope":scope,
                "qubits":m["qubits"],
                "depth":m["depth"],
                "size":m["size"],
                "oneq":m["oneq"],
                "twoq":m["twoq"],
                "multiq":m["multiq"],
                "twoq_depth":m["twoq_depth"],
                "build_time_s":item["build_time_s"],
                "transpile_time_s":item["transpile_time_s"],
                "estimated_qpu_us":item["estimated_qpu_us"],
            })

            for gate,count in m["ops"].items():
                gate_rows.append({
                    "transform":name,
                    "scope":scope,
                    "gate":gate,
                    "count":int(count),
                })

        # -------------------------------------------------------------
        # Circuit drawings
        # -------------------------------------------------------------
        if SAVE_CIRCUIT_DRAWINGS:
            save_block_circuit_drawing(transform, circuits_dir/f"{name.lower()}_block.png", name)
            save_expanded_circuit_drawing(transform, circuits_dir/f"{name.lower()}_expanded.png", reps=2)
            save_native_circuit_drawing(transform, circuits_dir/f"{name.lower()}_native_basis.png", basis_gates=RESOURCE_BASIS, opt_level=RESOURCE_OPT_LEVEL)
            save_native_circuit_drawing(transform, circuits_dir/f"{name.lower()}_fake_sherbrooke.png", backend=fake, opt_level=RESOURCE_OPT_LEVEL)

            if SAVE_DETAILED_CIRCUITS:
                save_detailed_circuit_drawing(
                    transform,
                    circuits_dir/f"{name.lower()}_detailed.png",
                )

    resources=pd.DataFrame(rows)
    gates=pd.DataFrame(gate_rows)

    resources.to_csv(
        output_dir/"cw_quantum_circuit_resources.csv",
        index=False,
    )

    gates.to_csv(
        output_dir/"cw_quantum_gate_counts.csv",
        index=False,
    )

    save_quantum_resource_plots(
        resources,
        gates,
        plots_dir,
    )

    print(
        "Saved quantum circuit resources: "
        f"{(output_dir/'cw_quantum_circuit_resources.csv').resolve()}"
    )

    return resources, gates

def save_quantum_resource_plots(resources,gates,plots_dir):
    plots_dir=Path(plots_dir)
    plots_dir.mkdir(parents=True,exist_ok=True)

    scope_order=[
        "transform_only",
        "stateprep_transform",
        "FakeSherbrooke",
    ]

    # 27 — Circuit depth
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=resources,
        x="transform",
        y="depth",
        hue="scope",
        hue_order=scope_order,
    )
    plt.xlabel("Quantum transform")
    plt.ylabel("Circuit depth")
    plt.title("Quantum Circuit Depth")
    plt.tight_layout()
    plt.savefig(
        plots_dir/"27_quantum_resource_depth.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close()

    # 28 — Total circuit size
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=resources,
        x="transform",
        y="size",
        hue="scope",
        hue_order=scope_order,
    )
    plt.xlabel("Quantum transform")
    plt.ylabel("Circuit operations")
    plt.title("Quantum Circuit Size")
    plt.tight_layout()
    plt.savefig(
        plots_dir/"28_quantum_resource_size.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close()

    # 29 — Two-qubit operation count
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=resources,
        x="transform",
        y="twoq",
        hue="scope",
        hue_order=scope_order,
    )
    plt.xlabel("Quantum transform")
    plt.ylabel("Two-qubit operations")
    plt.title("Quantum Two-Qubit Gate Cost")
    plt.tight_layout()
    plt.savefig(
        plots_dir/"29_quantum_resource_twoq.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close()

    # 30 — Two-qubit depth
    twoq=resources.dropna(subset=["twoq_depth"])

    if not twoq.empty:
        plt.figure(figsize=(10,6))
        sns.barplot(
            data=twoq,
            x="transform",
            y="twoq_depth",
            hue="scope",
            hue_order=scope_order,
        )
        plt.xlabel("Quantum transform")
        plt.ylabel("Two-qubit depth")
        plt.title("Quantum Two-Qubit Circuit Depth")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"30_quantum_resource_twoq_depth.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

    # 31 — Required qubits
    transform_only=resources[
        resources["scope"]=="transform_only"
    ]

    if not transform_only.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(
            data=transform_only,
            x="transform",
            y="qubits",
        )
        plt.xlabel("Quantum transform")
        plt.ylabel("Qubits")
        plt.title("Quantum Transform Qubit Requirements")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"31_quantum_resource_qubits.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

    # 32 — FakeSherbrooke native gate composition heatmap
    device_gates=gates[
        gates["scope"]=="FakeSherbrooke"
    ]

    if not device_gates.empty:
        heat=device_gates.pivot_table(
            index="transform",
            columns="gate",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )

        plt.figure(
            figsize=(
                max(9,1.1*len(heat.columns)),
                5,
            )
        )
        sns.heatmap(
            heat,
            annot=True,
            fmt=".0f",
        )
        plt.xlabel("Native / transpiled operation")
        plt.ylabel("Quantum transform")
        plt.title("FakeSherbrooke Gate Composition")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"32_quantum_resource_gate_heatmap.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

    # 33 — StatePreparation depth overhead
    depth=resources.pivot_table(
        index="transform",
        columns="scope",
        values="depth",
        aggfunc="mean",
    )

    if {
        "transform_only",
        "stateprep_transform",
    }.issubset(depth.columns):

        overhead=(
            depth["stateprep_transform"]
            / depth["transform_only"]
        )

        plt.figure(figsize=(8,5))
        plt.bar(
            overhead.index,
            overhead.values,
        )
        plt.xlabel("Quantum transform")
        plt.ylabel("Depth overhead ratio")
        plt.title("StatePreparation Circuit-Depth Overhead")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"33_stateprep_depth_overhead.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

    # 34 — Device mapping depth overhead
    if {
        "stateprep_transform",
        "FakeSherbrooke",
    }.issubset(depth.columns):

        overhead=(
            depth["FakeSherbrooke"]
            / depth["stateprep_transform"]
        )

        plt.figure(figsize=(8,5))
        plt.bar(
            overhead.index,
            overhead.values,
        )
        plt.xlabel("Quantum transform")
        plt.ylabel("Depth overhead ratio")
        plt.title("FakeSherbrooke Mapping Depth Overhead")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"34_device_mapping_depth_overhead.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

    # 37 — Circuit construction time
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=resources,
        x="transform",
        y="build_time_s",
        hue="scope",
    )
    plt.xlabel("Quantum transform")
    plt.ylabel("Build time (s)")
    plt.title("Quantum Circuit Construction Time")
    plt.tight_layout()
    plt.savefig(
        plots_dir/"37_quantum_resource_build_time.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close()

    # 38 — Transpilation time
    plt.figure(figsize=(10,6))
    sns.barplot(
        data=resources,
        x="transform",
        y="transpile_time_s",
        hue="scope",
    )
    plt.xlabel("Quantum transform")
    plt.ylabel("Transpilation time (s)")
    plt.title("Quantum Circuit Transpilation Time")
    plt.tight_layout()
    plt.savefig(
        plots_dir/"38_quantum_resource_transpile_time.png",
        dpi=350,
        bbox_inches="tight",
    )
    plt.close()

    # 39 — Estimated QPU execution duration
    device=resources[
        (resources["scope"]=="FakeSherbrooke")
        & resources["estimated_qpu_us"].notna()
    ]

    if not device.empty:
        plt.figure(figsize=(9,6))
        sns.barplot(
            data=device,
            x="transform",
            y="estimated_qpu_us",
        )
        plt.xlabel("Quantum transform")
        plt.ylabel("Estimated execution duration (µs)")
        plt.title("Estimated FakeSherbrooke Circuit Duration")
        plt.tight_layout()
        plt.savefig(
            plots_dir/"39_quantum_resource_qpu_duration.png",
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()

def run_noise_analysis(noise_metadata,output_csv):
    fake=FakeSherbrooke()
    sherbrooke_noise=NoiseModel.from_backend(fake)

    noise_simulators={
        "ideal":AerSimulator(
            method="density_matrix",
        ),
        "FakeSherbrooke":AerSimulator(
            method="density_matrix",
            noise_model=sherbrooke_noise,
        ),
    }

    noisy_transforms={
        "QFT":cached_iqft,
        "QDCT":cached_qdct,
        "QDST":cached_qdst,
        "QDWT":cached_qhaar,
    }

    def select_noise_window_indices(clean_signal):
        windows,_,_=split_windows(clean_signal)
        energies=np.linalg.norm(windows,axis=1)
        valid=np.flatnonzero(energies>1e-15)

        if len(valid)==0:
            return np.array([],dtype=int)

        if NOISE_MAX_WINDOWS is None or len(valid)<=NOISE_MAX_WINDOWS:
            return valid

        positions=np.linspace(
            0,
            len(valid)-1,
            NOISE_MAX_WINDOWS,
            dtype=int,
        )

        return valid[positions]

    def run_noisy_transform(x,transform,simulator):
        qc,norm,n=build_transform_pipeline(x,transform)

        if qc is None:
            return {
                "fidelity":1.0,
                "leakage":0.0,
                "depth":0,
                "size":0,
                "ops":{},
                "simulation_time_s":0.0,
            }

        tqc=transpile(
            qc,
            simulator,
            optimization_level=1,
        )

        ideal=Statevector.from_instruction(tqc)

        noisy_qc=tqc.copy()
        noisy_qc.save_density_matrix()

        t0=perf_counter()
        result=simulator.run(noisy_qc).result()
        simulation_time=perf_counter()-t0

        rho=DensityMatrix(
            result.data(0)["density_matrix"]
        )

        fidelity=float(
            state_fidelity(
                ideal,
                rho,
            )
        )

        leakage=0.0

        if transform.num_qubits==n+1:
            N=2**n
            diagonal=np.real(
                np.diag(rho.data)
            )

            leakage=max(
                0.0,
                1.0-float(
                    np.sum(diagonal[:N])
                ),
            )

        return {
            "fidelity":fidelity,
            "leakage":leakage,
            "depth":tqc.depth(),
            "size":tqc.size(),
            "ops":dict(tqc.count_ops()),
            "simulation_time_s":simulation_time,
        }

    def noise_analyze_recording(
        signal,
        window_indices,
        simulator,
        device_name,
        signal_type,
        row,
    ):
        windows,_,_=split_windows(signal)

        window_indices=np.asarray(
            window_indices,
            dtype=int,
        )

        windows=windows[window_indices]

        rows=[]

        print(
            f"    {signal_type}: "
            f"{len(windows)} windows"
        )

        for name,builder in noisy_transforms.items():
            fidelities=[]
            leakages=[]
            depths=[]
            sizes=[]
            twoq=[]
            simulation_times=[]

            print(
                f"      {name}: 0/{len(windows)}",
                end="",
                flush=True,
            )

            for wi,window in enumerate(windows,1):
                if np.linalg.norm(window)<=1e-15:
                    continue

                n=int(np.log2(len(window)))

                m=run_noisy_transform(
                    window,
                    builder(n),
                    simulator,
                )

                fidelities.append(
                    m["fidelity"]
                )
                leakages.append(
                    m["leakage"]
                )
                depths.append(
                    m["depth"]
                )
                sizes.append(
                    m["size"]
                )
                simulation_times.append(
                    m["simulation_time_s"]
                )

                ops=(
                    m["ops"]
                    if isinstance(m["ops"],dict)
                    else dict(m["ops"])
                )

                twoq_count=sum(
                    ops.get(g,0)
                    for g in [
                        "cx",
                        "cz",
                        "cp",
                        "swap",
                        "ecr",
                    ]
                )

                twoq.append(
                    twoq_count
                )

                if wi%25==0 or wi==len(windows):
                    print(
                        f"\r      {name}: "
                        f"{wi}/{len(windows)}",
                        end="",
                        flush=True,
                    )

            print()

            rows.append({
                "message_id":row["message_id"],
                "wpm":row["wpm"],
                "snr_requested_db":(
                    np.nan
                    if signal_type=="clean"
                    else row["snr_requested_db"]
                ),
                "signal_type":signal_type,
                "device":device_name,
                "transform":name,
                "windows_available":len(windows),
                "windows_tested":len(fidelities),
                "fidelity_mean":float(np.mean(fidelities)) if fidelities else np.nan,
                "fidelity_std":float(np.std(fidelities)) if fidelities else np.nan,
                "fidelity_min":float(np.min(fidelities)) if fidelities else np.nan,
                "leakage_mean":float(np.mean(leakages)) if leakages else np.nan,
                "leakage_max":float(np.max(leakages)) if leakages else np.nan,
                "depth_mean":float(np.mean(depths)) if depths else np.nan,
                "depth_std":float(np.std(depths)) if depths else np.nan,
                "size_mean":float(np.mean(sizes)) if sizes else np.nan,
                "size_std":float(np.std(sizes)) if sizes else np.nan,
                "twoq_mean":float(np.mean(twoq)) if twoq else np.nan,
                "twoq_std":float(np.std(twoq)) if twoq else np.nan,
                "simulation_time_mean_s":float(np.mean(simulation_times)) if simulation_times else np.nan,
                "simulation_time_std_s":float(np.std(simulation_times)) if simulation_times else np.nan,
                "simulation_time_total_s":float(np.sum(simulation_times)) if simulation_times else np.nan,
            })

        return pd.DataFrame(rows)

    all_noise_results=[]
    clean_done=set()

    total_rows=len(noise_metadata)

    for ri,(_,row) in enumerate(
        noise_metadata.iterrows(),
        1,
    ):
        sample=load_cw_sample(row)

        clean_id=(
            row["message_id"],
            row["wpm"],
        )

        include_clean=(
            clean_id not in clean_done
        )

        window_indices=select_noise_window_indices(
            sample["clean"]
        )

        if len(window_indices)==0:
            print(
                f"Skipping msg={row['message_id']} "
                f"wpm={row['wpm']}: no valid windows"
            )
            continue

        print(
            f"Noise recording "
            f"[{ri}/{total_rows}] "
            f"msg={row['message_id']} "
            f"wpm={row['wpm']} "
            f"snr={row['snr_requested_db']} "
            f"windows={len(window_indices)}"
        )

        signal_cases=[
            (
                "noisy",
                sample["noisy"],
            ),
        ]

        if include_clean:
            signal_cases.insert(
                0,
                (
                    "clean",
                    sample["clean"],
                ),
            )

        for device_name,simulator in noise_simulators.items():
            print(
                f"  Noise device: {device_name}"
            )

            for signal_type,signal in signal_cases:
                d=noise_analyze_recording(
                    signal,
                    window_indices,
                    simulator,
                    device_name,
                    signal_type,
                    row,
                )

                all_noise_results.append(d)

        if include_clean:
            clean_done.add(clean_id)

    if not all_noise_results:
        raise ValueError(
            "No quantum noise results were generated."
        )

    noise_results=pd.concat(
        all_noise_results,
        ignore_index=True,
    )

    output_csv=Path(output_csv)

    noise_results.to_csv(
        output_csv,
        index=False,
    )

    save_noise_plots(
        noise_results,
        output_csv.parent/"plots",
    )

    print(
        f"Saved noise results: "
        f"{output_csv.resolve()}"
    )

    return noise_results


def configure_paths(dataset):
    global CW_DATASET, CLEAN_DIR, NOISY_DIR, ENVELOPE_DIR, AUDIO_DIR, AUDIO_CLEAN_DIR, AUDIO_NOISY_DIR, METADATA_FILE
    CW_DATASET = Path(dataset).resolve()
    CLEAN_DIR = CW_DATASET / "clean"
    NOISY_DIR = CW_DATASET / "noisy"
    ENVELOPE_DIR = CW_DATASET / "envelopes"
    AUDIO_DIR = CW_DATASET / "audio"
    AUDIO_CLEAN_DIR = AUDIO_DIR / "clean"
    AUDIO_NOISY_DIR = AUDIO_DIR / "noisy"
    METADATA_FILE = CW_DATASET / "metadata.csv"


def main():
    global CPU_QSIM, STATEPREP_EXECUTOR
    configure_paths(CW_DATASET)
    CPU_QSIM = AerSimulator(method="statevector", device="CPU", max_parallel_threads=0, max_parallel_experiments=0)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_FILE}")

    output_dir=OUTPUT_DIR.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    metadata=pd.read_csv(METADATA_FILE)

    quantum_message_ids=metadata["message_id"].drop_duplicates().iloc[:QUANTUM_MESSAGE_COUNT].tolist()
    quantum_metadata=metadata[
        metadata["message_id"].isin(quantum_message_ids)
        & (metadata["wpm"]==QUANTUM_WPM)
        & metadata["snr_requested_db"].isin(QUANTUM_SNRS)
    ].copy()

    if MODE=="quantum":
        experiment_metadata=quantum_metadata
        transforms=BATCH_TRANSFORMS
        result_name="cw_quantum_subset_results.csv"
        reconstructed_name="cw_quantum_subset_reconstructed.npz"
    elif MODE=="classical":
        # Full classical benchmark only.
        experiment_metadata=metadata
        transforms=CLASSICAL_BATCH_TRANSFORMS
        result_name="cw_classical_results.csv"
        reconstructed_name="cw_classical_reconstructed.npz"
    elif MODE=="all":
        # Full dataset, classical + quantum.
        experiment_metadata=metadata
        transforms=BATCH_TRANSFORMS
        result_name="cw_compression_full_results.csv"
        reconstructed_name="cw_reconstructed_signals.npz"

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    if experiment_metadata.empty:
        raise ValueError("Experiment metadata selection is empty.")

    print(f"Mode: {MODE}")
    print(f"Rows: {len(experiment_metadata)}")
    print(f"StatePreparation workers: {QISKIT_STATEPREP_PROCS}")
    print(f"Transpile processes: {QISKIT_TRANSPILE_PROCS}")
    print(f"Quantum Aer batch size: {Q_BATCH_SIZE}")
    print(f"Dataset: {CW_DATASET}")
    print(f"Output: {output_dir}")

    all_results, all_reconstructed, clean_done = [], {}, set()

    executor = None
    try:
        if MODE in {"quantum","all"} and QISKIT_STATEPREP_PROCS > 1:
            # One pool for the whole experiment: workers are spawned once and reused.
            executor = ProcessPoolExecutor(
                max_workers=QISKIT_STATEPREP_PROCS,
                mp_context=mp.get_context("spawn"),
            )
            STATEPREP_EXECUTOR = executor

        for i, (_, row) in enumerate(experiment_metadata.iterrows()):
            sample = load_cw_sample(row)
            clean_id = (row["message_id"], row["wpm"])
            include_clean = clean_id not in clean_done
            if include_clean:
                clean_done.add(clean_id)

            result_df, reconstructed_signals = run_cw_experiment(
                sample["clean"], sample["noisy"], row, transforms, include_clean
            )
            all_results.append(result_df)

            if SAVE_RECONSTRUCTED:
                for (signal_type, transform, retention), waveform in reconstructed_signals.items():
                    if signal_type == "clean":
                        key = f"msg{int(row['message_id']):03d}_wpm{int(row['wpm']):02d}_clean_{transform}_r{int(round(retention * 100)):03d}"
                    else:
                        key = f"msg{int(row['message_id']):03d}_wpm{int(row['wpm']):02d}_snr{int(row['snr_requested_db']):+d}_{transform}_r{int(round(retention * 100)):03d}"
                    all_reconstructed[key] = waveform

            print(f"[{i + 1}/{len(experiment_metadata)}] complete", flush=True)
    finally:
        STATEPREP_EXECUTOR = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    full_results = pd.concat(all_results, ignore_index=True)
    full_results["retention_pct"] = full_results["retention_requested"] * 100
    result_path = output_dir / result_name
    full_results.to_csv(result_path, index=False)
    print(f"Saved results: {result_path}")

    if SAVE_RECONSTRUCTED:
        reconstructed_path = output_dir / reconstructed_name
        np.savez_compressed(reconstructed_path, **all_reconstructed)
        print(f"Saved reconstructed waveforms: {reconstructed_path}")

    if SAVE_PLOTS:
        save_plots(full_results, output_dir / "plots")

    if SAVE_CIRCUIT_RESOURCES and MODE in {"quantum","all"}:
        row=experiment_metadata.iloc[0]
        sample=load_cw_sample(row)
        windows,_,_=split_windows(sample["clean"])

        nonzero_windows=[
            window
            for window in windows
            if np.linalg.norm(window)>1e-15
        ]

        if not nonzero_windows:
            raise ValueError(
                "No nonzero window available for circuit resource analysis."
            )

        representative_window=nonzero_windows[
            len(nonzero_windows)//2
        ]

        save_quantum_circuit_resources(
            representative_window,
            output_dir,
        )

    if RUN_NOISE:
        run_noise_analysis(
            quantum_metadata,
            output_dir/"cw_quantum_noise_results.csv",
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
