## ADDED Requirements

### Requirement: Shop-Integrated WandB Logger
The system SHALL provide a wandb logger config that uses shop's WandB utilities to avoid config field conflicts.

#### Scenario: Logger instantiation without conflict
- **WHEN** user specifies `logger=wandb_shop`
- **THEN** WandB run is created without "multiple values for argument 'config'" error

### Requirement: Foundation Encoder Experiment Config
The system SHALL provide experiment configs for foundation model encoders that work with manylatents.main.

#### Scenario: Central dogma fusion experiment runs
- **WHEN** user runs `python -m manylatents.main --config-name=config experiment=central_dogma_fusion`
- **THEN** all three encoders (Evo2, Orthrus, ESM3) load and produce 3840-dim embeddings

### Requirement: Optional Cluster Dependency
The system SHALL make shop an optional dependency under `[cluster]` extras.

#### Scenario: Local execution without shop
- **WHEN** user installs without `[cluster]` extras
- **THEN** local execution works but cluster submission is unavailable

#### Scenario: Cluster submission with shop
- **WHEN** user installs with `pip install .[cluster]`
- **THEN** cluster submission via `cluster=mila_remote` works

### Requirement: Dogma Extras Installation via Wheelnext
The system SHALL document and support installation of dogma extras using wheelnext uv.

#### Scenario: Fresh dogma installation
- **GIVEN** user has installed wheelnext uv
- **WHEN** user runs `uv lock --index-strategy unsafe-best-match && uv sync --extra dogma --index-strategy unsafe-best-match`
- **THEN** all dogma dependencies are installed from prebuilt wheels

#### Scenario: Import verification on login node
- **GIVEN** dogma extras are installed
- **WHEN** user runs `module load cuda/12.4.1 && uv run python -c "import evo2, orthrus, esm"`
- **THEN** all imports succeed without errors

#### Scenario: GPU execution via SLURM
- **GIVEN** dogma extras are installed
- **WHEN** job is submitted to GPU node via submitit
- **THEN** encoders load and execute without CUDA module load (CUDA auto-available on GPU nodes)

### Requirement: Version Compatibility
The system SHALL pin package versions for ABI compatibility.

#### Scenario: torch version constraint
- **GIVEN** transformer-engine-torch is built against torch 2.7.x
- **WHEN** lockfile is generated
- **THEN** torch version is constrained to >=2.7,<2.8

#### Scenario: torchvision version constraint
- **GIVEN** torchvision must match torch major version
- **WHEN** lockfile is generated
- **THEN** torchvision version is constrained to >=0.22,<0.23

### Requirement: mamba-ssm 2.x Compatibility
The system SHALL maintain compatibility with mamba-ssm 2.x API changes.

#### Scenario: Block import shim
- **GIVEN** mamba-ssm 2.x moved Block class to new location
- **WHEN** OrthrusEncoder is loaded
- **THEN** Block class is available at expected import path via compatibility shim
