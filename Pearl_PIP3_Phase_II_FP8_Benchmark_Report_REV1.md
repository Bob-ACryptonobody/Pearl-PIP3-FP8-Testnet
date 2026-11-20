# Pearl PIP-3 | Phase II

Standalone FP8 Kernel Benchmark — Test Plan, Execution History, Results and Code Appendices

Pearl-Diver-3 • NVIDIA RTX 5070 Ti 16 GB • Blackwell SM120 • November 20, 2026

*Document revision REV1  |  Beta experimental study  |  Prepared for GitHub review*

Status: Four sustained workload tests completed. Kernel execution and synthetic protocol-equivalent throughput demonstrated. Network-valid mining NOT demonstrated.

## Executive summary

Phase I combined Qwen2.5-3B-Instruct inference and Pearl’s FP8 mining integration on the same RTX 5070 Ti, producing roughly 1.0–1.2 TH/s of protocol-credited activity under selected inference workloads. Because the shared model and GPU workload limited the ability to characterize standalone kernel performance, Phase II isolated the GPU and executed Pearl’s original mixed_gemm() implementation directly using synthetic FP8 matrices without an LLM.

The central question was whether low measured performance reflected a GPU limitation or simply insufficient work submitted per kernel launch. Three increasing eligible workload shapes and a final doubling experiment were run for 60 seconds each with unchanged GPU power and clocks. Throughput rose from 43.29 to 85.71 to 146.01 TH/s. Doubling the largest workload again produced only 146.52 TH/s, indicating a plateau under these tested conditions.

Every TH/s figure in Phase II is SYNTHETIC PROTOCOL-EQUIVALENT COMPUTATIONAL THROUGHPUT. It is not the confirmed accepted-share rate, does not represent distinct new cryptographic jobs, and cannot establish stand-alone network mining without an LLM.

### Document map

1. Purpose and assumptions  •  2. Hardware and software configuration  •  3. Calculation method  •  4. Chronological execution and decision log  •  5. Results and interpretation  •  6. Limitations, replication and next steps  •  Appendices A–L: command/code stages in execution order.

## 1. Purpose, questions and pre-test assumptions

Why: Test the Pearl FP8 compute path independent of AI inference; reproduce the order of magnitude of third-party RTX 5070 Ti standalone benchmark reports if the methodology is comparable; establish a controlled workload and hardware baseline for later power and architecture studies.

- Can the original Pearl mixed_gemm() kernel execute with synthetic matrices and no loaded LLM?
- How much eligible M×N×K work per invocation is needed to approach full reported GPU utilization?
- How does workload geometry change throughput, board power, clocks and thermal conditions?
- Can a repeatable calculation of normalized TH/s be traced back to Pearl source and tests?
- What cannot be inferred about mining proofs, gateway credit or actual network results?
### Working assumptions and scope

- Synthetic matrix operands are useful for measuring repeatable kernel computational performance, but cannot substitute for realistic proof-producing jobs.
- Pearl’s committed Blackwell tile selector is the eligibility authority for work accounting. PyTorch device(cuda:0) is not the same as miner_base.commitment.Device.BLACKWELL.
- The four main workload tests change matrix dimensions only; benchmark implementation, power cap, and manual clock settings remain unchanged.
- GPU utilization is an NVIDIA device-wide sampling metric, not a direct Tensor Core occupancy measurement.
- Work accounting is normalized to equivalent hashes, not actual numbers of accepted proofs or submitted shares.
- External reports of approximately 150–156 TH/s are contextual only until matched source code, test settings, and timing are established.
## 2. Hardware and software configuration

| Component | Configuration / evidence |
| --- | --- |
| Host | Pearl-Diver-3 (PD3), Pop!_OS Linux, username bob, host pop-os |
| Motherboard / CPU | Biostar TB360-BTC Pro 2.0 / Intel Pentium Gold G5400T |
| GPU 1 / benchmark target | NVIDIA GeForce RTX 5070 Ti 16 GB (16,303 MiB visible); CUDA capability (12,0) / SM120 |
| GPU 1 UUID | GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2 |
| Other GPUs | GPU 0 RTX 5070 12GB; GPU 2 RTX 4060 Ti; left on production SRBMiner |
| Power | Existing RTX 5070 Ti power limit 300.00 W; no adjustments in Phase II |
| Power system | Dual-PSU PD3 configuration, CyberPower UPS; study uses GPU board telemetry, not measured wall energy |
| Miner service | pearl-miner.service active, temporary ExecStart isolation --gpu-id 0,2 |
| Pearl development branch | fp8-scheme, reported local revision 1dd75a52e073b4a67079c7b6758228d71865ccea |
| Python environment | /home/bob/pearl-fp8-test/pearl/.venv/bin/python |
| Kernel | pearl_gemm.mixed_gemm / default_mixed_gemm_config / record_hits=False |
| Torch version | 2.13.0+cu130 as reported in test environment |
| Telemetry | nvidia-smi samples approximately once per second; output CSV + summary JSON |
| Artifacts | /home/bob/pearl-fp8-test/standalone-bench/REV1, REV2, REV3 and /home/bob/Archive/Pearl-FP8-Experiments/ |

*Provenance note: settings are the observed existing configuration, not an independent confirmation that all BIOS, driver, GPU boost and clock parameters matched factory stock. The model path and full package lockfile for Phase I are not part of the Phase II code annex.*

## 3. What “TH/s” means in this study

A matrix operation has dimensions M rows, N output columns and K common (inner) dimension. Pearl’s effective_work_per_matmul() uses a committed architecture-specific lottery tile and returns difficulty-normalized work. For eligible shapes the tile geometry cancels, giving exactly M × N × K work units per launch.

Equivalent TH/s = (M × N × K × completed kernel launches) / (elapsed wall-clock seconds × 1,000,000,000,000).

Example, Test D: 4096 × 14336 × 4096 = 240,518,168,576 equivalent work units per launch. In 60.094416 seconds, 36,608 launches completed, giving 8,804,889,115,230,208 total equivalent units and 146.5176 TH/s. A standard GEMM FLOPS convention counts approximately 2 × M × N × K floating-point operations; these FLOPS and Pearl-equivalent TH/s are different units.

A kernel may execute correctly with an unmineable shape (e.g. K=512). Only shapes accepted by select_tile()/effective_work_per_matmul() were used in sustained TH/s tests. No network threshold, valid proof generation, gateway credit, or independent unique attempts were established. Inputs, pow_key, and other fields remained fixed; hit recording was off.

## 4. Chronological execution and decision log

| Step | Activity / output | Decision and reason |
| --- | --- | --- |
| 0 | Prior Phase I Qwen + Pearl integration: ~1.0–1.2 credited TH/s in selected windows | Separate inference and kernel ceiling; do not compare rates as the same workload. |
| 1 | Production isolation: --gpu-id 0,2; RTX 5070 Ti free | Protect ongoing mining on GPUs 0 and 2; run synthetic experiment only on GPU 1. |
| 2 | REV1 small kernel smoke: 256×128×512; 20 launches; 0.007894 s; 0.0850 mathematical TFLOPS | Kernel works without LLM; throughput not a valid credited rate because shape fails mining eligibility. |
| 3 | Source audit found effective_work_per_matmul(); discovered committed tiles | Replace FLOPS-only reporting with Pearl’s own normalized work calculation. |
| 4 | Device diagnosis: torch.device(cuda:0) rejected; use Device.BLACKWELL | Correct the device enum; eligible shapes then passed. |
| 5 | REV2: 1024×4096×4096; 20 launches; 42.8818 equivalent TH/s | Valid eligible shape, but 8 ms run too short for utilization and power analysis. |
| 6 | REV2 timing audit: synchronize before/after batch; launch overhead included | Build REV3 with 60-second windows, CUDA events and NVIDIA telemetry. |
| 7 | REV3 Test A: 43.2866 TH/s, 127.66 W, 28.67% utilization | Underworked; double M. |
| 8 | REV3 Test B: 85.7079 TH/s, 207.96 W, 56.48% utilization | Workload increased use; increase output N. |
| 9 | REV3 Test C: 146.0147 TH/s, 295.39 W, 98.48% utilization | Nearly full utilization/power; double M once to test plateau. |
| 10 | REV3 Test D: 146.5176 TH/s, 295.36 W, 97.86% utilization | Only +0.34%; stop enlarging for this configuration; next study power curve. |

Interpretation of near-identical C and D average board watts: both approached the configured 300 W limit. The observation is consistent with a power/compute ceiling but does not, without throttle reason telemetry, prove precisely which GPU subsystem is limiting throughput.

## 5. Measured performance and supporting calculations

| Test | M × N × K | Work/launch | Launches | Elapsed (s) | TH/s | Board W | Util. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1024×4096×4096 | 17,179,869,184 | 151,296 | 60.0474 | 43.2866 | 127.66 | 28.67% |
| B | 2048×4096×4096 | 34,359,738,368 | 149,760 | 60.0378 | 85.7079 | 207.96 | 56.48% |
| C | 2048×14336×4096 | 120,259,084,288 | 72,960 | 60.0906 | 146.0147 | 295.39 | 98.48% |
| D | 4096×14336×4096 | 240,518,168,576 | 36,608 | 60.0944 | 146.5176 | 295.36 | 97.86% |

| Test | CUDA-event equivalent TH/s | TH/s per avg board W | Mean temp | Max temp | Mean graphics clock | VRAM used |
| --- | --- | --- | --- | --- | --- | --- |
| A | 43.3305 | 0.3391 | 51.95 °C | 56 °C | 2813.95 MHz | 510 MiB |
| B | 85.7928 | 0.4121 | 60.67 °C | 64 °C | 2749.17 MHz | 540 MiB |
| C | 146.1256 | 0.4943 | 66.48 °C | 69 °C | 2392.84 MHz | 860 MiB |
| D | 146.5876 | 0.4961 | 67.26 °C | 69 °C | 2389.03 MHz | 924 MiB |

Each REV3 test returned 58 telemetry samples and zero telemetry errors. Memory clock was reported as 13,801 MHz throughout the loaded measurement windows. Board power averaged from telemetry includes low-power startup/finish samples: the published TH/s/W figures are preliminary estimates and should not be represented as steady-state electrical efficiency.

Derived changes: A→B equivalent TH/s +98.0%; B→C +70.4%; C→D +0.344%. Estimated TH/s/W improves ~46% A→D. Test D has almost exactly twice C’s work per launch but about half the launch rate; total equivalent work completed in a minute scarcely changes.

## 6. Limitations, reproduction rules and next work

- No LLM, no fresh keys or job construction, no proof publication, no network-valid shares; this is NOT evidence of a stand-alone PIP-3 miner.
- record_hits=False excludes win recording; kernel correctness/proof equivalence under real conditions remains for developer review.
- The code uses a zero threshold and fixed key; mathematically normalized throughput is an accounting proxy, not earned mining credit.
- 60-second tests and one-second nvidia-smi samples do not establish long-term stability, error/reject rates, or whole-system energy.
- The 300 W configured limit was retained; neither clock/power changes nor true stock settings were independently established.
- Comparable Rabid Mining performance numbers are context, NOT independent methodological validation.
- Exact source files and raw JSON/CSV are on PD3, not attached to this conversation. GitHub publication should include byte-for-byte exports, hashes and dependencies before claims of reproducibility.
- Next Phase III: explore separate inference host and mine-capable GPU integration, if supported by Pearl; record actual gateway credited work, proofs, inference latency, and power.
### Recommended repository layout

```bash
Phase-II-FP8/
  README.md                 # Link to this report and experiment scope
  docs/Phase-II-FP8-Report.docx
  src/REV1/fp8_bench_rev1.py
  src/REV2/fp8_bench_rev2.py
  src/REV3/fp8_bench_rev3.py
  results/Test-A/{results.json,gpu-telemetry.csv,console.log}
  results/Test-B/{results.json,gpu-telemetry.csv,console.log}
  results/Test-C/{results.json,gpu-telemetry.csv,console.log}
  results/Test-D/{results.json,gpu-telemetry.csv,console.log}
  environment/{commit.txt,sha256sums.txt,packages.txt,gpu-info.txt}
```

Source-integrity record: REV1 SHA256 f857a24931257ba723a8abaaa97d1dcce7405543e4c5ba12de5e00136396a824; REV2 bb05b6c8122613816cdcbe6c989e0e2ea7d48676521d030102c74f5db1ddbcbb; REV3 4228cf985e2129f1c5c3122edc2336273b2df7c5df8f1b61231cef3765d896b7.

## Appendix A — GPU isolation and REV1 baseline

A temporary systemd override limited production SRBMiner to GPUs 0 and 2. The original miner unit was backed up. The RTX 5070 Ti UUID was the dedicated synthetic-test target. The full executed override creation command and full REV1 source have not been pasted into the available conversation; they must be exported from PD3, not reconstructed as exact transcripts.

### Command / code recorded for this stage

```bash
# Read-only checks used throughout the study
systemctl is-active pearl-miner.service
SYSTEMD_PAGER=cat systemctl show -p ExecStart --value pearl-miner.service
nvidia-smi --query-gpu=index,name,uuid,memory.used --format=csv
 
# REV1 source to export from PD3:
# /home/bob/pearl-fp8-test/standalone-bench/REV1/fp8_bench_rev1.py
# Verified SHA256:
# f857a24931257ba723a8abaaa97d1dcce7405543e4c5ba12de5e00136396a824
```

### Observed output / evidence

The production service was active with --gpu-id 0,2. Small REV1 smoke test: 256×128×512; 20 iterations; elapsed 0.007893584 s; 2,533.7 calls/s; mathematical 0.08501698 TFLOPS. Output: /home/bob/Archive/Pearl-FP8-Experiments/standalone-rev1-20261120-084803/results.json.

### Why this led to the next stage

Confirmed a stand-alone mixed_gemm() call worked on the isolated SM120 GPU. Did not claim Pearl TH/s pending source-level accounting and workload eligibility checks.

## Appendix B — Find official work-accounting code

The initial search pinpointed credited-work functions, telemetry and validation tests. This step contained no GPU execution.

### Command / code recorded for this stage

```bash
cd ~/pearl-fp8-test/pearl || exit 1
# Locate work accounting and published telemetry fields.
grep -RInE \
 'effective_work_per_matmul|total_credited|delta_credited|interval_ths|average_ths' \
 miner --include='*.py' --exclude-dir=__pycache__ | head -65
# Search for accounting terms in miner implementations.
grep -RInE \
 'credited_work|effective_work|work_per|hashrate|hash_rate' \
 miner/miner-base/src miner/vllm-miner/src --include='*.py' | head -65
systemctl is-active pearl-miner.service
```

### Observed output / evidence

Found miner/vllm-miner/src/vllm_miner/mining_config.py:288 effective_work_per_matmul(), pipeline.py:365 credit integration, and async_loop_manager.py:427 reported interval_ths/average_ths.

### Why this led to the next stage

Decided to use Pearl’s code for normalized work rather than labeling FLOPS as TH/s.

## Appendix C — Inspect formula, pipeline and unit tests

The source and tests explain that in-bound committed lottery tiles allow difficulty-normalized M×N×K equivalent work. Tile selection prevents invalid shapes from receiving credit.

### Command / code recorded for this stage

```bash
cd ~/pearl-fp8-test/pearl || exit 1
# Review the actual formula and its rejection rules.
sed -n '260,325p' miner/vllm-miner/src/vllm_miner/mining_config.py
# See how the serving pipeline credits launches.
sed -n '345,390p' miner/vllm-miner/src/vllm_miner/pipeline.py
# Examine independent test assertions.
sed -n '145,195p' miner/vllm-miner/tests/test_mining_config.py
systemctl is-active pearl-miner.service
```

### Observed output / evidence

effective_work_per_matmul(m,n,k,device) chooses a committed tile and returns lottery_hashes_per_matmul(m,n,tile) × tile.rows × tile.cols × k = M×N×K where eligible. The verifier refuses K<1024; K=512 is not eligible.

### Why this led to the next stage

REV1’s K=512 was shown to be a kernel smoke test, not an eligible credited-work throughput test.

## Appendix D — Diagnose architecture-specific device argument

The first validation mistakenly passed torch.device("cuda:0") to a function that expects Pearl’s Device enum. The architecture was checked, then the committed tile selector inspected.

### Command / code recorded for this stage

```bash
cd ~/pearl-fp8-test/pearl || exit 1
# Inspect Blackwell/Hopper committed tile policy.
sed -n '95,165p' miner/vllm-miner/src/vllm_miner/mining_config.py
# Confirm SM120 architecture (read-only).
CUDA_VISIBLE_DEVICES=GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2 \
 .venv/bin/python -c \
 'import torch; print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))'
systemctl is-active pearl-miner.service
```

### Observed output / evidence

Device.BLACKWELL has committed tiles 4×64, 16×32 and 4×128. Device was CUDA capability (12,0). Previously, passing torch.device gave “no committed lottery tiles for device(type='cuda', index=0)”.

### Why this led to the next stage

Validated the correct distinction: `torch.device` selects a CUDA runtime target; `Device.BLACKWELL` selects Pearl’s protocol architecture policy.

## Appendix E — Confirm eligible dimensions with Device.BLACKWELL

The second eligibility run used Pearl’s enum and checked the exact candidate matrix dimensions before running kernels.

### Command / code recorded for this stage

```bash
cd ~/pearl-fp8-test/pearl || exit 1
CUDA_VISIBLE_DEVICES=GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2 \
.venv/bin/python - <<'PYCODE'
from miner_base.commitment import Device
from vllm_miner.mining_config import effective_work_per_matmul, select_tile
for m,n,k in [(256,128,512),(1024,4096,4096),
              (2048,4096,4096),(2048,14336,4096)]:
    try:
        tile=select_tile(n,k,device=Device.BLACKWELL)
        work=effective_work_per_matmul(m,n,k,device=Device.BLACKWELL)
        print('PASS',m,n,k,'tile=',tile,'work=',work)
    except (ValueError,TypeError) as exc:
        print('FAIL',m,n,k,exc)
PYCODE
systemctl is-active pearl-miner.service
```

### Observed output / evidence

256×128×512 FAIL. 1024×4096×4096 PASS 17,179,869,184. 2048×4096×4096 PASS 34,359,738,368. 2048×14336×4096 PASS 120,259,084,288. All passing shapes selected tile 4×64.

### Why this led to the next stage

Selected the smallest eligible shape as the controlled REV2 starting point.

## Appendix F — REV2 build and eligible 20-launch smoke test

REV2 was derived from the verified REV1 file without changing Pearl’s kernel. The construction command was a guarded text-transform; the complete byte-for-byte REV2 source should accompany this document in GitHub. The excerpt below shows the critical changes and the execution arguments.

### Command / code recorded for this stage

```bash
# Versioned creation: preserve REV1; checksum validation.
SOURCE="$HOME/pearl-fp8-test/standalone-bench/REV1/fp8_bench_rev1.py"
DEST="$HOME/pearl-fp8-test/standalone-bench/REV2/fp8_bench_rev2.py"
echo 'f857a24931257ba723a8abaaa97d1dcce7405543e4c5ba12de5e00136396a824  '"$SOURCE" | sha256sum -c -
# Key REV2 additions (source excerpt, NOT a complete executable file):
# from miner_base.commitment import Device
# from vllm_miner.mining_config import effective_work_per_matmul
# work_per_launch = effective_work_per_matmul(m,n,k,device=Device.BLACKWELL)
# protocol_equivalent_ths = calls_per_second * work_per_launch / 1e12
# Baseline eligible workload: M=1024, N=4096, K=4096, iterations=20.
# Run (only after GPU isolation is independently checked):
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2 \
"$HOME/pearl-fp8-test/pearl/.venv/bin/python" \
"$DEST" --run --m 1024 --n 4096 --k 4096 --iterations 20 \
--output "$HOME/Archive/Pearl-FP8-Experiments/standalone-rev2-20261120-091949/results.json"
```

### Observed output / evidence

REV2 SHA256 bb05b6c8122613816cdcbe6c989e0e2ea7d48676521d030102c74f5db1ddbcbb; 20 launches, elapsed 0.008012656 s; 2,496.05 calls/s; 42.88183387 equivalent TH/s, 85.7637 mathematical TFLOPS. No proof/credit recorded.

### Why this led to the next stage

Demonstrated the accounted workload. Eight milliseconds was too short to estimate sustained power and temperature, prompting timing audit and REV3.

## Appendix G — REV2 timing audit and REV3 instrument design

Verified that warmup and compile were outside the timed section and that torch.cuda.synchronize() completed queued work before stopping the timer. REV3 retained the kernel setup and added duration-based batches, event timing and a separate NVIDIA telemetry thread.

### Command / code recorded for this stage

```bash
BENCH="$HOME/pearl-fp8-test/standalone-bench/REV2/fp8_bench_rev2.py"
grep -nE 'def launch|warmup|perf_counter|synchronize|iterations|calls_per_second|mixed_gemm' "$BENCH"
sed -n '1,260p' "$BENCH"
systemctl is-active pearl-miner.service
 
# Relevant observed REV2 timing loop (source excerpt):
# start = time.perf_counter()
# for _ in range(args.iterations):
#     launch()          # calls pearl_gemm.mixed_gemm(..., record_hits=False)
# torch.cuda.synchronize()
# elapsed = time.perf_counter() - start
# calls_per_second = args.iterations / elapsed
 
# REV3's key new measurement logic (excerpt):
# while time.perf_counter() - wall_start < args.duration_seconds:
#     start_event = torch.cuda.Event(enable_timing=True)
#     stop_event  = torch.cuda.Event(enable_timing=True)
#     start_event.record()
#     for _ in range(args.batch_size): launch()
#     stop_event.record(); stop_event.synchronize()
#     total_cuda_ms += start_event.elapsed_time(stop_event)
#     total_launches += args.batch_size
```

### Observed output / evidence

REV3 syntax PASS; config-only mode 1024×4096×4096, 60 seconds, batch size 128; REV3 SHA256 4228cf985e2129f1c5c3122edc2336273b2df7c5df8f1b61231cef3765d896b7. Source verified before each test.

### Why this led to the next stage

Move to sustained repeated measurements; preserve a second timing clock and capture utilization/power/clocks/VRAM in CSV. Source file in repository should be authoritative, rather than retyped snippets.

## Appendix H — REV3 Test A: 1024×4096×4096

Test A was conducted using the same versioned REV3 program, 300 W configured power limit, unchanged clocks, 60-second duration and 128 launches per batch. The displayed command is the reproducible execution core with the recorded output path. The originally pasted preflight shell also checked GPU index→UUID and available VRAM.

### Command / code recorded for this stage

```bash
# REV3-BETA TEST A: execute AFTER confirming production isolation.
# The persistent GPU UUID prevents selecting the wrong card by index.
BENCH="$HOME/pearl-fp8-test/standalone-bench/REV3/fp8_bench_rev3.py"
GPU="GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2"
EXPECTED="4228cf985e2129f1c5c3122edc2336273b2df7c5df8f1b61231cef3765d896b7"
echo "$EXPECTED  $BENCH" | sha256sum -c -
systemctl is-active --quiet pearl-miner.service || exit 1
SYSTEMD_PAGER=cat systemctl show -p ExecStart --value pearl-miner.service \
 | grep -q -- '--gpu-id 0,2' || exit 1
# Ensure GPU 1 has minimal used VRAM before running.
nvidia-smi -i "$GPU" --query-gpu=uuid,memory.used,power.limit --format=csv
# Saved output directory from the successful recorded execution:
RESULTS="$HOME/Archive/Pearl-FP8-Experiments/standalone-rev3-20261120-093143"
# Executed workload with explicit duration and batch size.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
 "$HOME/pearl-fp8-test/pearl/.venv/bin/python" "$BENCH" \
 --run --duration-seconds 60 --batch-size 128 \
 --m 1024 --n 4096 --k 4096 \
 --output "$RESULTS/results.json" 2>&1 | tee "$RESULTS/console.log"
# Review completion and primary telemetry.
python3 -m json.tool "$RESULTS/results.json" | head -65
systemctl is-active pearl-miner.service
```

### Observed output / evidence

Elapsed 60.047390 s; 151,296 launches; host protocol-equivalent 43.286569 TH/s; average reported board power 127.656 W; average NVIDIA utilization 28.67%; zero telemetry errors; 58 telemetry samples. Data: standalone-rev3-20261120-093143/results.json and gpu-telemetry.csv.

### Why this led to the next stage

Utilization only 28.7%; double activation rows to Test B.

## Appendix I — REV3 Test B: 2048×4096×4096

Test B was conducted using the same versioned REV3 program, 300 W configured power limit, unchanged clocks, 60-second duration and 128 launches per batch. The displayed command is the reproducible execution core with the recorded output path. The originally pasted preflight shell also checked GPU index→UUID and available VRAM.

### Command / code recorded for this stage

```bash
# REV3-BETA TEST B: execute AFTER confirming production isolation.
# The persistent GPU UUID prevents selecting the wrong card by index.
BENCH="$HOME/pearl-fp8-test/standalone-bench/REV3/fp8_bench_rev3.py"
GPU="GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2"
EXPECTED="4228cf985e2129f1c5c3122edc2336273b2df7c5df8f1b61231cef3765d896b7"
echo "$EXPECTED  $BENCH" | sha256sum -c -
systemctl is-active --quiet pearl-miner.service || exit 1
SYSTEMD_PAGER=cat systemctl show -p ExecStart --value pearl-miner.service \
 | grep -q -- '--gpu-id 0,2' || exit 1
# Ensure GPU 1 has minimal used VRAM before running.
nvidia-smi -i "$GPU" --query-gpu=uuid,memory.used,power.limit --format=csv
# Saved output directory from the successful recorded execution:
RESULTS="$HOME/Archive/Pearl-FP8-Experiments/standalone-rev3-testB-0qcxWJkV"
# Executed workload with explicit duration and batch size.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
 "$HOME/pearl-fp8-test/pearl/.venv/bin/python" "$BENCH" \
 --run --duration-seconds 60 --batch-size 128 \
 --m 2048 --n 4096 --k 4096 \
 --output "$RESULTS/results.json" 2>&1 | tee "$RESULTS/console.log"
# Review completion and primary telemetry.
python3 -m json.tool "$RESULTS/results.json" | head -65
systemctl is-active pearl-miner.service
```

### Observed output / evidence

Elapsed 60.037835 s; 149,760 launches; host protocol-equivalent 85.707861 TH/s; average reported board power 207.959 W; average NVIDIA utilization 56.48%; zero telemetry errors; 58 telemetry samples. Data: standalone-rev3-testB-0qcxWJkV/results.json and gpu-telemetry.csv.

### Why this led to the next stage

Equivalent throughput nearly doubled with similar launches per second; increase N for Test C.

## Appendix J — REV3 Test C: 2048×14336×4096

Test C was conducted using the same versioned REV3 program, 300 W configured power limit, unchanged clocks, 60-second duration and 128 launches per batch. The displayed command is the reproducible execution core with the recorded output path. The originally pasted preflight shell also checked GPU index→UUID and available VRAM.

### Command / code recorded for this stage

```bash
# REV3-BETA TEST C: execute AFTER confirming production isolation.
# The persistent GPU UUID prevents selecting the wrong card by index.
BENCH="$HOME/pearl-fp8-test/standalone-bench/REV3/fp8_bench_rev3.py"
GPU="GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2"
EXPECTED="4228cf985e2129f1c5c3122edc2336273b2df7c5df8f1b61231cef3765d896b7"
echo "$EXPECTED  $BENCH" | sha256sum -c -
systemctl is-active --quiet pearl-miner.service || exit 1
SYSTEMD_PAGER=cat systemctl show -p ExecStart --value pearl-miner.service \
 | grep -q -- '--gpu-id 0,2' || exit 1
# Ensure GPU 1 has minimal used VRAM before running.
nvidia-smi -i "$GPU" --query-gpu=uuid,memory.used,power.limit --format=csv
# Saved output directory from the successful recorded execution:
RESULTS="$HOME/Archive/Pearl-FP8-Experiments/standalone-rev3-testC-4gM45gO1"
# Executed workload with explicit duration and batch size.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
 "$HOME/pearl-fp8-test/pearl/.venv/bin/python" "$BENCH" \
 --run --duration-seconds 60 --batch-size 128 \
 --m 2048 --n 14336 --k 4096 \
 --output "$RESULTS/results.json" 2>&1 | tee "$RESULTS/console.log"
# Review completion and primary telemetry.
python3 -m json.tool "$RESULTS/results.json" | head -65
systemctl is-active pearl-miner.service
```

### Observed output / evidence

Elapsed 60.090554 s; 72,960 launches; host protocol-equivalent 146.014676 TH/s; average reported board power 295.387 W; average NVIDIA utilization 98.48%; zero telemetry errors; 58 telemetry samples. Data: standalone-rev3-testC-4gM45gO1/results.json and gpu-telemetry.csv.

### Why this led to the next stage

Near-full utilization and board power; double M once more to determine whether a plateau has formed.

## Appendix K — REV3 Test D: 4096×14336×4096

Test D was conducted using the same versioned REV3 program, 300 W configured power limit, unchanged clocks, 60-second duration and 128 launches per batch. The displayed command is the reproducible execution core with the recorded output path. The originally pasted preflight shell also checked GPU index→UUID and available VRAM.

### Command / code recorded for this stage

```bash
# REV3-BETA TEST D: execute AFTER confirming production isolation.
# The persistent GPU UUID prevents selecting the wrong card by index.
BENCH="$HOME/pearl-fp8-test/standalone-bench/REV3/fp8_bench_rev3.py"
GPU="GPU-0f480801-9f39-ecf0-4d8b-ed0d9eeefdf2"
EXPECTED="4228cf985e2129f1c5c3122edc2336273b2df7c5df8f1b61231cef3765d896b7"
echo "$EXPECTED  $BENCH" | sha256sum -c -
systemctl is-active --quiet pearl-miner.service || exit 1
SYSTEMD_PAGER=cat systemctl show -p ExecStart --value pearl-miner.service \
 | grep -q -- '--gpu-id 0,2' || exit 1
# Ensure GPU 1 has minimal used VRAM before running.
nvidia-smi -i "$GPU" --query-gpu=uuid,memory.used,power.limit --format=csv
# Saved output directory from the successful recorded execution:
RESULTS="$HOME/Archive/Pearl-FP8-Experiments/standalone-rev3-testD-6fZ1bt8N"
# Executed workload with explicit duration and batch size.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
 "$HOME/pearl-fp8-test/pearl/.venv/bin/python" "$BENCH" \
 --run --duration-seconds 60 --batch-size 128 \
 --m 4096 --n 14336 --k 4096 \
 --output "$RESULTS/results.json" 2>&1 | tee "$RESULTS/console.log"
# Review completion and primary telemetry.
python3 -m json.tool "$RESULTS/results.json" | head -65
systemctl is-active pearl-miner.service
```

### Observed output / evidence

Elapsed 60.094416 s; 36,608 launches; host protocol-equivalent 146.517591 TH/s; average reported board power 295.356 W; average NVIDIA utilization 97.86%; zero telemetry errors; 58 telemetry samples. Data: standalone-rev3-testD-6fZ1bt8N/results.json and gpu-telemetry.csv.

### Why this led to the next stage

Only 0.34% gain vs C, indicating no practical benefit from this larger matrix at the tested power limit.

## Appendix L — Audit/export commands for the GitHub repository

Use these commands on PD3 to export authoritative sources and logs. The Word report contains the study, excerpts and full test execution chronology, but does not substitute for the actual .py files and raw CSV/JSON already preserved on PD3. The commands below are proposed for archival; they are NOT claimed as already executed.

### Command / code recorded for this stage

```bash
# READ-ONLY export inventory; run on PD3.
ROOT="$HOME/pearl-fp8-test/standalone-bench"
for revision in REV1 REV2 REV3; do
  find "$ROOT/$revision" -maxdepth 1 -type f -name '*.py' \
    -exec sha256sum {} \
    -exec ls -lh {} \;
done
# Locate every saved Phase II run (JSON, CSV and logs).
find "$HOME/Archive/Pearl-FP8-Experiments" -maxdepth 2 \
 -type f \( -name 'results.json' -o -name 'console.log' \
 -o -name 'gpu-telemetry.csv' \) -print
# Capture the actual source revision and Python environment.
git -C "$HOME/pearl-fp8-test/pearl" rev-parse HEAD
"$HOME/pearl-fp8-test/pearl/.venv/bin/python" -m pip freeze
```

### Observed output / evidence

Known version checksums (REV1/REV2/REV3) are preserved in the report. Real source/telemetry archives reside on PD3, not in the document-construction environment.

### Why this led to the next stage

Publishing raw files and checksums makes this report independently auditable. Avoid uploading personal wallet credentials, tokens, keys, or private home-network configuration.

*End of report  •  Experimental beta  •  No network-valid mining claims  •  Independent review requested*
