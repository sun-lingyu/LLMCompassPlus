# Before use the scripts in this folder

## Prepare and compile FA2/3

**Install FA3**

Clone and install FA3 from github.
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git apply target_scripts/sm87_customization.patch # On Orin
export MAX_JOBS=4 && export NVCC_THREADS=1
cd hopper && python ./setup.py bdist_wheel
cd dist && pip install flash_attn_3-3.0.0-cp39-abi3-linux_aarch64.whl
```

FA3 can be imported from `flash_attn_interface`.

**Install FA2**

Download pre-built binary from https://pypi.jetson-ai-lab.io/jp6/cu126 for Orin.

FA2 can be imported from `flash_attn`.