# Pearl-PIP3-FP8-Testnet
Testing Documentation for proposed PIP 3/FP8 update software on  personal GPUS
# Pearl PIP-3 FP8 Testnet — Independent GPU Testing

**Project Status:** BETA — Experimental Testing  
**Date:** November 2026  
**Platform:** Pearl Research Labs — PIP-3 FP8 Testnet2  
**Test System:** Pearl-Diver-3
*** Concept and Local Testing Architecture by: Bob Lynch AKA "Bob-A-Crypto-Nobody", Coding by ChatGPT 
> **IMPORTANT — EXPERIMENTAL SOFTWARE — USE AT YOUR OWN RISK**
>
> This is an independent, community-driven testing and documentation project. It is not an official Pearl Research Labs publication.
>
> Review and understand all code, scripts, installation commands, and third-party dependencies before executing them. Do not blindly copy and execute commands.
>
> Hardware configurations, operating systems, drivers, firmware, and software dependencies vary. Procedures that worked on the test system may not work on yours.
>
> All software and documentation are provided **AS IS**, without warranties or guarantees. Anyone choosing to use these procedures assumes responsibility for potential hardware damage, data loss, security vulnerabilities, service interruptions, financial losses, or other consequences.
>
> Back up important data and conduct experiments in an isolated environment whenever possible.

---

## 1. Why This Project Exists

Pearl Research Labs is developing PIP-3, a change to its proof-of-work system involving FP8 (8-bit floating-point) matrix computations.

FP8 operations are commonly associated with modern AI accelerators and GPU Tensor Cores.

The proposed changes raise several practical questions for existing GPU miners:

- Can consumer NVIDIA GPUs participate in PIP-3 FP8 mining?
- Which GPU architectures are compatible?
- What software, drivers, and dependencies are required?
- What happens to GPU performance and power consumption?
- Can FP8 mining operate reliably on an ordinary Linux mining rig?
- Will the network accept the work produced by these GPUs?

The objective is to investigate these questions through independent, repeatable experiments using actual hardware.

It was found that setting up an independent FP8 test environment requires a number of software components that may be unfamiliar to ordinary GPU miners.

I decided to document the entire process so other interested individuals can establish their own test environments, reproduce the tests, and share findings.
Writing





A secondary objective of this project is to explore how AI can serve as a force multiplier, helping individuals become more effective and efficient when investigating complex technical problems.

AI is being used to assist with research, coding, test development, analysis, and documentation. The intent is not to replace technical expertise or independent verification, but to make those capabilities more accessible and productive.  

**This project is intended to encourage testing, transparency, technical review, and constructive community feedback.**

## 2. Test Hardware — Pearl-Diver-3

The initial test system is an existing GPU mining rig named **Pearl-Diver-3 (PD3)**.

### System Configuration

| Component | Hardware / Software |
|---|---|
| Motherboard | Biostar TB360-BTC PRO 2.0 |
| Processor | Intel Pentium Gold G5400T |
| Operating System | Pop!_OS 24.04 LTS |
| System RAM | 15 GiB |
| GPU 1 | NVIDIA GeForce RTX 5070 — 12 GB |
| GPU 2 | NVIDIA GeForce RTX 5070 Ti — 16 GB |
| GPU 3 | NVIDIA GeForce RTX 4060 Ti — 16 GB |
| NVIDIA Driver | 595.104.02 |
| CUDA Toolkit | 13.0 |
| Go Compiler | 1.26.10 |
| Rust / Cargo | 1.100.0 |
| Python Environment | Python 3.12 with `uv` |
| Power Supplies | Dual Corsair power supplies |
| Existing Production Miner | SRBMiner-MULTI |

**Important:** These specifications describe the equipment used for testing. They are **not official minimum system requirements** for Pearl PIP-3.

Other configurations may require different drivers, dependencies, CUDA versions, and installation procedures.

### Initial GPU Under Test

The  first FP8 experiment uses the:

**NVIDIA GeForce RTX 5070 Ti — 16 GB VRAM**

This is an NVIDIA Blackwell-generation GPU with compute capability SM120.

The RTX 5070 and RTX 4060 Ti remain part of the test system but are not included in the initial two-hour FP8 mining experiment.

**Additional GPUs will be evaluated in subsequent testing**, subject to software compatibility and availability.

The eventual goal is to compare GPUs across different NVIDIA generations, looking at compatibility, performance, power consumption, temperature, and stability.

## 3. Testing Approach

Rather than immediately replacing production mining software or modifying an operational rig, I adopted an incremental approach.

The objectives were to:

1. Establish a separate development and test environment.
2. Obtain the experimental Pearl FP8 source code.
3. Install the required build and GPU computing dependencies.
4. Verify FP8 computational functionality using a single GPU.
5. Compile and operate a real Pearl testnet2 blockchain node.
6. Synchronize the node with the public test network.
7. Integrate the compatible FP8 mining software.
8. Conduct a controlled two-hour mining experiment.
9. Document the results and identify compatibility or stability issues.

Where possible, existing production mining operations are preserved while individual components are tested.

This minimizes unnecessary disruption and helps isolate problems when they occur.

## 4. Work Completed So Far

### FP8 Development Environment

The experimental Pearl source code was downloaded from the official Pearl Research Labs GitHub repository, using its `fp8-scheme` development branch.

The software was installed and verified the required software components, including:

- NVIDIA CUDA Toolkit
- Python and the `uv` package manager
- PyTorch and Pearl GPU computation dependencies
- NVIDIA CUTLASS-related libraries
- Rust and Cargo
- Go compiler and Go Task build utility
- GCC, Make, and supporting development tools

The specific source revision and software versions used in the initial experiment are recorded in the detailed setup documentation.

### RTX 5070 Ti — Preliminary FP8 Testing

The software successfully executed FP8 matrix multiplication on the RTX 5070 Ti.

Selected tests from the Pearl FP8 source also passed, including tests for computational consistency and committed lottery verification behavior.

These preliminary results established that parts of the experimental FP8 software can execute successfully on the RTX 5070 Ti.

**This does not yet establish full PIP-3 miner compatibility or demonstrate that the GPU can submit network-accepted mining work.**

### Pearl Testnet2 Blockchain Node

Successfully compiled the experimental Pearl blockchain node and its command-line control utility:

- `pearld`
- `prlctl`

The node reports version `1.6.0-fp8-testnet1`.

It was started with separate testnet2 storage and configuration paths to avoid interfering with an existing mainnet environment.

The node began downloading and validating blockchain data from the public Pearl test network.

At the latest documented checkpoint, the node had reached **block 150,487**, beyond the previously reported PIP-3 activation height of 150,283.

At that checkpoint:

- Blockchain storage occupied approximately 8.6 GB.
- The node process remained active.
- Production SRBMiner remained active.
- Final synchronization and network-tip verification had not yet been documented.

This is an important intermediate result, but it is not the completion of the GPU mining test.

## 5. Planned Two-Hour FP8 Mining Experiment

Once the testnet2 node has synchronized and the GPU mining software is operational, I intend to perform a controlled experiment using one RTX 5070 Ti.

| Parameter | Initial Test Plan |
|---|---|
| GPU | RTX 5070 Ti 16 GB |
| Power limit | 300 watts |
| Test network | Pearl PIP-3 Testnet2 |
| Stabilization | 2 minutes |
| Measurement period | 120 minutes |
| Other GPUs | Continue production mining where possible |

The measurement period begins only after the node is synchronized, the FP8 mining software is operational, and the stabilization period is complete.

I intend to collect GPU work rate, average power consumption, GPU temperature, errors, rejected work, and independently verifiable evidence of accepted mining work where available.

The test also requires a safe cutoff and restoration procedure so the GPU can return to normal production mining.

**The two-hour network mining test has not yet been completed. No final performance conclusions are being claimed at this stage.**

## 6. Installation and Setup Documentation

A detailed guide has been prepared for other interested testers.

**[Pearl PIP-3 FP8 Testnet2 — Independent Test Environment (PDF)](PEARL_PIP3_Testnet2_Field_Guide_REV1.pdf)**

**[Editable Markdown Setup Guide](PEARL_PIP3_Testnet2_Field_Guide_REV1.md)**

The documentation covers software prerequisites, installation steps, source compilation, preliminary FP8 checks, testnet2 node configuration, synchronization monitoring, and safety considerations.

It also explains the purpose of the commands and distinguishes verified procedures from work that remains experimental.

**Before following the guide, compare every requirement and command against your own operating system, hardware, and the current official Pearl source code.**

The instructions are based on a specific development revision and may require changes as PIP-3 evolves.

## 7. Planned Additional GPU Testing

After the initial RTX 5070 Ti experiment, I will to expand testing to other available GPUs.

Potential candidates include the RTX 5070 and RTX 4060 Ti already installed in Pearl-Diver-3, as well as additional NVIDIA GPUs available on other systems.

Not every GPU will necessarily support the required FP8 operations or the specific Pearl mining implementation.

I will distinguish clearly between:

- Hardware that supports an FP8 numerical format.
- Hardware that passes selected computational tests.
- Hardware that successfully runs the complete Pearl miner.
- Hardware that produces independently verified, network-accepted mining work.

These are different levels of compatibility and should not be confused.

## 8. Results and Reports

**Results will be published as testing progresses.**

Planned reporting includes GPU compatibility observations, mining performance, power consumption, average temperatures, software errors, rejected work, operational stability, and other relevant measurements.

I intend to document failed tests and unsuccessful configurations, as these may be just as valuable to other developers and miners as successful results.

All performance measurements should include sufficient information about the hardware, software versions, test duration, and methodology to support meaningful comparison.

## 9. Community Participation

Independent testing and constructive technical feedback are welcome.

If you reproduce these experiments, please include your GPU model, operating system, driver and CUDA versions, Pearl source revision, installation changes, test methodology, and observed results.

Please use GitHub Issues to report bugs, compatibility problems, documentation corrections, or suggestions.

Do not publish private keys, wallet seed phrases, authentication credentials, sensitive network details, or other confidential information.

## 10. Project Independence and Credits

This is an independent experimental project and is **not affiliated with, endorsed by, or maintained by Pearl Research Labs**.

Pearl Research Labs is responsible for its own protocol and software development. This repository documents independent experiences with that software.

Project direction, hardware experimentation, testing, and observations are contributed by the project maintainer, with ChatGPT assistance in technical analysis and documentation.

Additional contributors and reviewers will be acknowledged as appropriate.

---

### Final Notice

**BETA — FOR EXPERIMENTAL AND EDUCATIONAL USE ONLY**

Nothing in this repository constitutes a guarantee of compatibility, mining performance, financial return, hardware safety, or network acceptance.

Review all code before execution. Test responsibly. Maintain backups. Protect production systems.

**Use entirely at your own risk.**
