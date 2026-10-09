import numpy as np
from pathlib import Path
from scipy.io.wavfile import write

NPZ_PATH=Path("cw_outputs/cw_quantum_subset_reconstructed.npz")
OUTPUT_DIR=Path("cw_reconstructed_audio")
FS=8000

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
data=np.load(NPZ_PATH,allow_pickle=True)

for name in data.files:
    x=np.asarray(data[name],dtype=np.float64)
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
    peak=np.max(np.abs(x))
    if peak>0:
        x=x/peak
    wav=np.int16(np.clip(x,-1.0,1.0)*32767)
    write(OUTPUT_DIR/f"{name}.wav",FS,wav)

print(f"Saved {len(data.files)} WAV files to {OUTPUT_DIR}")