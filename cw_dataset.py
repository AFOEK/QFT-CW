import numpy as np
from pathlib import Path
from scipy.io import wavfile
import csv

WPMS = [5, 10, 15, 20, 25, 30]
SNRS_DB = [40, 30, 20, 10, 5, 0, -5, -10]

SAMPLE_RATE = 8000
TONE_HZ = 700
RISE_FALL_MS = 5.0

SEED = 42
METADATA = []

MORSE_CODES = {
    "CQ CQ CQ DE VA3FMU K":
        "-.-. --.- / -.-. --.- / -.-. --.- / -.. . / ...- .- ...-- ..-. -- ..- / -.-",

    "CQ DX CQ DX DE VA3FMU K":
        "-.-. --.- / -.. -..- / -.-. --.- / -.. -..- / -.. . / ...- .- ...-- ..-. -- ..- / -.-",

    "CQ TEST DE VA3FMU VA3FMU":
        "-.-. --.- / - . ... - / -.. . / ...- .- ...-- ..-. -- ..- / ...- .- ...-- ..-. -- ..-",

    "VA3PARC DE VA3FMU VA3FMU K":
        "...- .- ...-- .--. .- .-. -.-. / -.. . / ...- .- ...-- ..-. -- ..- / ...- .- ...-- ..-. -- ..- / -.-",

    "VA3PARC DE VA3FMU GE TNX FER CALL UR RST 579 579 QTH ONCA ONCA K":
        "...- .- ...-- .--. .- .-. -.-. / -.. . / ...- .- ...-- ..-. -- ..- / --. . / - -. -..- / ..-. . .-. / -.-. .- .-.. .-.. / ..- .-. / .-. ... - / ..... --... ----. / ..... --... ----. / --.- - .... / --- -. -.-. .- / --- -. -.-. .- / -.-",

    "SOS SOS SOS":
        "...---... / ...---... / ...---...",

    "VA3PARC DE VA3FMU RR TNX ALL OK HR":
        "...- .- ...-- .--. .- .-. -.-. / -.. . / ...- .- ...-- ..-. -- ..- / .-. .-. / - -. -..- / .- .-.. .-.. / --- -.- / .... .-.",

    "VA3PARC DE VA3FMU RR FB TNX QSO HPE CU AGN 73 ES GUD DX VA3PARC DE VA3FMU <SK>":
        "...- .- ...-- .--. .- .-. -.-. / -.. . / ...- .- ...-- ..-. -- ..- / .-. .-. / ..-. -... / - -. -..- / --.- ... --- / .... .--. . / -.-. ..- / .- --. -. / --... ...-- / . ... / --. ..- -.. / -.. -..- / ...- .- ...-- .--. .- .-. -.-. / -.. . / ...- .- ...-- ..-. -- ..- / ...-.-",
}

OUTPUT_DIR = Path("cw_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_CLEAN_DIR = AUDIO_DIR / "clean"
AUDIO_NOISY_DIR = AUDIO_DIR / "noisy"
CLEAN_DIR = OUTPUT_DIR / "clean"
NOISY_DIR = OUTPUT_DIR / "noisy"
ENVELOPE_DIR = OUTPUT_DIR / "envelopes"

for directory in [CLEAN_DIR, NOISY_DIR, ENVELOPE_DIR, AUDIO_DIR, AUDIO_CLEAN_DIR, AUDIO_NOISY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def morse_to_envelope(morse, wpm, sample_rate=8000):
    unit_samples = round((1.2 / wpm) * sample_rate)
    def on(units):
        return np.ones(unit_samples * units, dtype=np.float32)

    def off(units):
        return np.zeros(unit_samples * units, dtype=np.float32)

    chunks = []
    words = morse.strip().split(" / ")
    for word_i, word in enumerate(words):
        letters = word.split()
        for letter_i, letter in enumerate(letters):
            for symbol_i, symbol in enumerate(letter):
                if symbol == ".":
                    chunks.append(on(1))
                elif symbol == "-":
                    chunks.append(on(3))
                else:
                    raise ValueError(f"Invalid Morse symbol: {symbol}")

                # gap between elements of same character
                if symbol_i < len(letter) - 1:
                    chunks.append(off(1))

            # gap between characters
            if letter_i < len(letters) - 1:
                chunks.append(off(3))

        # gap between words
        if word_i < len(words) - 1:
            chunks.append(off(7))

    return np.concatenate(chunks)

def shape_key_envelope(envelope, sample_rate=8000, rise_fall_ms=5.0):
    shaped = envelope.astype(np.float32).copy()
    ramp_samples = round(sample_rate * rise_fall_ms / 1000.0)
    # Locate transitions
    padded = np.pad(envelope, (1, 1))
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]

    for start, end in zip(starts, ends):
        pulse_length = end - start
        # Never let attack + decay consume the whole symbol
        n = min(ramp_samples, pulse_length // 2)
        if n < 2:
            continue

        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n)))
        shaped[start:start+n] *= ramp
        shaped[end-n:end] *= ramp[::-1]
    return shaped

def envelope_to_cw(envelope, tone_hz=700, sample_rate=8000):
    n = np.arange(len(envelope), dtype=np.float64)
    tone = np.sin(2 * np.pi * tone_hz * n / sample_rate)
    return (envelope * tone).astype(np.float32)

def add_awgn(clean, active_mask, snr_db, rng):
    active = active_mask > 0
    if not np.any(active):
        raise ValueError("Signal contains no active CW samples.")

    signal_power = np.mean(clean[active] ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=clean.shape,)
    return (clean + noise).astype(np.float32)

def save_wav(path, waveform, sample_rate=8000):
    peak = np.max(np.abs(waveform))
    if peak > 0:
        audio = waveform / peak
    else:
        audio = waveform
    audio = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sample_rate, (audio * 32767).astype(np.int16))

def measure_snr(clean, noisy, active_mask):
    active = active_mask > 0
    signal = clean[active]
    noise = noisy[active] - clean[active]
    ps = np.mean(signal ** 2)
    pn = np.mean(noise ** 2)
    return 10 * np.log10(ps / pn)

for morse_i, (text, morse) in enumerate(MORSE_CODES.items()):
    for wpm_i, wpm in enumerate(WPMS):
        hard = morse_to_envelope(morse, wpm, sample_rate=SAMPLE_RATE)
        shaped = shape_key_envelope(hard, sample_rate=SAMPLE_RATE, rise_fall_ms=RISE_FALL_MS)
        clean = envelope_to_cw(shaped, tone_hz=TONE_HZ, sample_rate=SAMPLE_RATE)
        save_wav(AUDIO_CLEAN_DIR / f"msg{morse_i:03d}_wpm{wpm:02d}_clean.wav", clean, SAMPLE_RATE)
        np.save(CLEAN_DIR / f"msg{morse_i:03d}_wpm{wpm:02d}_clean.npy", clean)
        np.save(ENVELOPE_DIR / f"msg{morse_i:03d}_wpm{wpm:02d}_hard.npy", hard)
        np.save(ENVELOPE_DIR / f"msg{morse_i:03d}_wpm{wpm:02d}_shaped.npy", shaped)
        for snr_i, snr in enumerate(SNRS_DB):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, morse_i, wpm_i, snr_i]))
            noisy = add_awgn(clean, hard, snr, rng)
            filename = (f"msg{morse_i:03d}_wpm{wpm:02d}_snr{snr:+03d}db_tone{TONE_HZ}.npy")
            np.save(NOISY_DIR / filename, noisy)
            wav_filename = (f"msg{morse_i:03d}_wpm{wpm:02d}_snr{snr:+03d}db_tone{TONE_HZ}.wav")
            save_wav(AUDIO_NOISY_DIR / wav_filename, noisy, SAMPLE_RATE,)
            actual = measure_snr(clean, noisy, hard)
            print(f"Requested={snr:+.1f} dB, Actual={actual:+.2f} dB")
            METADATA.append({
                "sample_id": f"msg{morse_i:03d}_wpm{wpm:02d}_snr{snr:+03d}db",

                "text": text,
                "morse": morse,

                "clean_file": f"msg{morse_i:03d}_wpm{wpm:02d}_clean.npy",
                "noisy_file": filename,
                "audio_file": wav_filename,

                "message_id": morse_i,
                "wpm": wpm,

                "snr_requested_db": snr,
                "snr_actual_db": actual,

                "tone_hz": TONE_HZ,
                "sample_rate": SAMPLE_RATE,
                "rise_fall_ms": RISE_FALL_MS,

                "noise_type": "AWGN",

                "seed_base": SEED,
                "seed_message": morse_i,
                "seed_wpm": wpm_i,
                "seed_snr": snr_i,
            })


with open(OUTPUT_DIR / "metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=METADATA[0].keys())
    writer.writeheader()
    writer.writerows(METADATA)