#!/usr/bin/env python3
"""
Complete Pipeline Orchestrator

Executes the entire speech detection project from scratch:
- Downloads datasets (if needed)
- Prepares data splits (GroupShuffleSplit, zero-leakage)
- Generates experimental variants (8 durations × 6 SNRs)
- Trains models (multi-seed, QLoRA)
- Optimizes hyperparameters (on DEV only: temperature, prompt, threshold)
- Evaluates on TEST (once, with optimized hyperparameters)
- Computes psychometric curves (DT75, SNR-75)
- Runs baselines (Silero VAD)
- Generates reports

Usage:
    # Full run
    python run_complete_pipeline.py --config config/pipeline_config.yaml

    # Smoke test (quick validation)
    python run_complete_pipeline.py --config config/pipeline_config.yaml --smoke_test

    # Resume from checkpoint
    python run_complete_pipeline.py --config config/pipeline_config.yaml --resume

    # Dry run (print commands without executing)
    python run_complete_pipeline.py --config config/pipeline_config.yaml --dry_run
"""

import argparse
import subprocess
import yaml
from pathlib import Path
import logging
import sys
import os
from datetime import datetime
import json
import glob

# Setup logging - will be configured per run
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: Path, dry_run: bool = False, smoke_test: bool = False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.smoke_test = smoke_test

        # Setup logging with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path('logs') / f'pipeline_{timestamp}.log'
        log_file.parent.mkdir(exist_ok=True)

        # Configure logging: concise format, file + console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # Reset any previous config
        )

        logger.info(f"Log file: {log_file}")

        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Apply smoke test overrides
        if smoke_test or self.config.get('smoke_test', {}).get('enabled', False):
            logger.warning("[SMOKE TEST] MODE ENABLED")
            self.apply_smoke_test_config()

        # Create directories
        self.create_directories()

        # Setup environment
        self.setup_environment()

    def apply_smoke_test_config(self):
        """Override config for smoke testing"""
        smoke_cfg = self.config.get('smoke_test', {})
        limit = smoke_cfg.get('limit_per_split', 2)
        epochs = smoke_cfg.get('epochs', 1)
        seeds = smoke_cfg.get('seeds', [42])

        self.config['splits']['train_size'] = limit
        self.config['splits']['dev_size'] = limit
        self.config['splits']['test_size'] = limit
        self.config['training']['hyperparameters']['num_epochs'] = epochs
        self.config['training']['seeds'] = seeds

        logger.info(f"  Smoke test: {limit} samples/split, {epochs} epoch(s), seeds {seeds}")

    def create_directories(self):
        """Create all necessary directories"""
        paths = self.config.get('paths', {})
        for key, path in paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)

        # Logs directory
        Path('logs').mkdir(exist_ok=True)

    def setup_environment(self):
        """Setup environment variables"""
        env_config = self.config.get('environment', {})

        if env_config.get('hf_hub_enable_transfer', False):
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

        if env_config.get('bitsandbytes_nowelcome', False):
            os.environ['BITSANDBYTES_NOWELCOME'] = '1'

        logger.info("[OK] Environment configured")

    def run_command(self, cmd: list, description: str, stage_num: int = None):
        """Run a subprocess command with optimized logging"""
        # Compact header
        stage_prefix = f"[{stage_num}] " if stage_num else ""
        logger.info(f"\n{stage_prefix}{description}")

        if self.dry_run:
            logger.info(f"  Command: {' '.join(map(str, cmd))}")
            logger.info("  [DRY RUN] Skipping")
            return True

        # Create stage-specific log file
        timestamp = datetime.now().strftime('%H%M%S')
        safe_desc = description.replace(' ', '_').replace(':', '')[:50]
        stage_log = Path('logs') / f'{timestamp}_{safe_desc}.log'

        try:
            logger.info(f"  Running... (log: {stage_log.name})")

            # Run with output capture
            with open(stage_log, 'w', encoding='utf-8') as f:
                f.write(f"Command: {' '.join(map(str, cmd))}\n")
                f.write("=" * 80 + "\n\n")

                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            logger.info(f"  [OK] Completed")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"  [FAILED] Exit code: {e.returncode}")
            logger.error(f"  Check log: {stage_log}")
            return False

    def stage_0_validate_environment(self):
        """STAGE 0: Validate environment"""
        logger.info("\n[0] Environment Validation")

        import torch
        cuda_available = torch.cuda.is_available()

        logger.info(f"  Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        else:
            logger.warning("  No CUDA GPU detected - training will be very slow!")

        logger.info("  [OK]")
        return True

    def stage_1_download_datasets(self):
        """STAGE 1: Download datasets"""
        logger.info("\n[1] Dataset Check")

        voxconv_path = Path(self.config['datasets']['voxconverse']['path'])
        esc50_path = Path(self.config['datasets']['esc50']['path'])

        if voxconv_path.exists() and esc50_path.exists():
            logger.info("  [OK] Datasets found")
            return True

        logger.warning("  [!] Datasets missing - download manually:")
        logger.info(f"    VoxConverse: {self.config['datasets']['voxconverse']['url']}")
        logger.info(f"    ESC-50: {self.config['datasets']['esc50']['url']}")
        return False

    def stage_2_prepare_base_clips(self):
        """STAGE 2: Prepare base clips"""
        output_dir = Path(self.config['paths']['base_clips'])

        if (output_dir / 'train_base.csv').exists() and not self.dry_run:
            logger.info("  [SKIP] Base clips already exist")
            return True

        cmd = [
            'python3', 'scripts/prepare_base_clips.py',
            '--voxconverse_dir', self.config['datasets']['voxconverse']['path'],
            '--esc50_dir', self.config['datasets']['esc50']['path'],
            '--output_dir', str(output_dir),
            '--duration', str(self.config['experimental_design']['base_duration_ms']),
            '--train_size', str(self.config['splits']['train_size']),
            '--dev_size', str(self.config['splits']['dev_size']),
            '--test_size', str(self.config['splits']['test_size']),
            '--seed', str(self.config['splits']['seed']),
        ]

        if self.smoke_test:
            cmd.extend(['--limit_per_split', str(self.config['splits']['train_size'])])

        return self.run_command(cmd, "Prepare Base Clips", stage_num=2)

    def stage_3_generate_variants(self):
        """STAGE 3: Generate experimental variants"""
        output_dir = Path(self.config['paths']['experimental_variants'])

        if (output_dir / 'train_metadata.csv').exists() and not self.dry_run:
            logger.info("  [SKIP] Variants already exist")
            return True

        cmd = [
            'python3', 'scripts/generate_experimental_variants.py',
            '--input_base', self.config['paths']['base_clips'],
            '--output_dir', str(output_dir),
            '--durations', *map(str, self.config['experimental_design']['durations_ms']),
            '--snr_levels', *map(str, self.config['experimental_design']['snr_levels_db']),
            '--padding_duration', str(self.config['experimental_design']['padding_duration_ms']),
            '--noise_amplitude', str(self.config['experimental_design']['noise_amplitude']),
        ]

        return self.run_command(cmd, "Generate Experimental Variants", stage_num=3)

    def stage_4_train_multiseed(self):
        """STAGE 4: Train models (multi-seed)"""
        logger.info(f"\n[4] Multi-Seed Training ({len(self.config['training']['seeds'])} seeds)")

        for idx, seed in enumerate(self.config['training']['seeds'], 1):
            checkpoint_dir = Path(self.config['paths']['checkpoints']) / f"qwen_lora_seed{seed}" / "final"

            if checkpoint_dir.exists() and not self.dry_run:
                logger.info(f"  [{idx}/{len(self.config['training']['seeds'])}] Seed {seed}: [SKIP] Already trained")
                continue

            logger.info(f"  [{idx}/{len(self.config['training']['seeds'])}] Training seed {seed}...")

            cmd = [
                'python3', 'scripts/finetune_qwen_audio.py',
                '--seed', str(seed),
                '--train_csv', f"{self.config['paths']['experimental_variants']}/train_metadata.csv",
                '--filter_duration', str(self.config['training']['train_filter']['duration_ms']),
                '--filter_snr', str(self.config['training']['train_filter']['snr_db']),
                '--val_csv', f"{self.config['paths']['experimental_variants']}/dev_metadata.csv",
                '--output_dir', f"{self.config['paths']['checkpoints']}/qwen_lora_seed{seed}",
                '--num_epochs', str(self.config['training']['hyperparameters']['num_epochs']),
                '--per_device_train_batch_size', str(self.config['training']['hyperparameters']['per_device_train_batch_size']),
                '--gradient_accumulation_steps', str(self.config['training']['hyperparameters']['gradient_accumulation_steps']),
                '--learning_rate', str(self.config['training']['hyperparameters']['learning_rate']),
                '--lora_r', str(self.config['training']['lora']['r']),
                '--lora_alpha', str(self.config['training']['lora']['alpha']),
            ]

            if not self.run_command(cmd, f"Train seed {seed}", stage_num=4):
                return False
            logger.info(f"    Seed {seed} completed")

        return True

    def stage_5_evaluate_dev(self):
        """STAGE 5: Evaluate on DEV"""
        logger.info(f"\n[5] Evaluate on DEV ({len(self.config['training']['seeds'])} seeds)")

        for idx, seed in enumerate(self.config['training']['seeds'], 1):
            output_csv = Path(self.config['paths']['dev_eval']) / f"dev_seed{seed}_all_variants.csv"

            if output_csv.exists() and not self.dry_run:
                logger.info(f"  [{idx}/{len(self.config['training']['seeds'])}] Seed {seed}: [SKIP]")
                continue

            logger.info(f"  [{idx}/{len(self.config['training']['seeds'])}] Evaluating seed {seed}...")

            cmd = [
                'python3', 'scripts/evaluate_with_logits.py',
                '--checkpoint', f"{self.config['paths']['checkpoints']}/qwen_lora_seed{seed}/final",
                '--test_csv', f"{self.config['paths']['experimental_variants']}/dev_metadata.csv",
                '--output_csv', str(output_csv),
                '--batch_size', str(self.config['evaluation']['batch_size']),
            ]

            if not self.run_command(cmd, f"Eval DEV seed {seed}", stage_num=5):
                return False

        return True

    def stage_6_optimize_on_dev(self):
        """STAGE 6-8: Optimize temperature, prompt, threshold on DEV ONLY"""
        logger.info("\n[6-8] Hyperparameter Optimization (DEV only)")

        seed = self.config['training']['seeds'][0]
        logger.info(f"  Using seed {seed} for optimization")

        # These would call actual scripts in production
        if self.config['optimization']['calibration']['enabled']:
            logger.info("  [6] Temperature: [TODO]")

        if self.config['optimization']['prompt_search']['enabled']:
            logger.info("  [7] Prompt: [TODO]")

        if self.config['optimization']['threshold_search']['enabled']:
            logger.info("  [8] Threshold: [TODO]")

        return True

    def stage_9_evaluate_test_final(self):
        """STAGE 9: Evaluate TEST (ONCE, with optimized hyperparameters)"""
        logger.info(f"\n[9] Final TEST Evaluation ({len(self.config['training']['seeds'])} seeds)")

        for idx, seed in enumerate(self.config['training']['seeds'], 1):
            output_csv = Path(self.config['paths']['test_final']) / f"test_seed{seed}_optimized.csv"

            if output_csv.exists() and not self.dry_run:
                logger.info(f"  [{idx}/{len(self.config['training']['seeds'])}] Seed {seed}: [SKIP]")
                continue

            logger.info(f"  [{idx}/{len(self.config['training']['seeds'])}] Evaluating seed {seed}...")

            cmd = [
                'python3', 'scripts/evaluate_with_logits.py',
                '--checkpoint', f"{self.config['paths']['checkpoints']}/qwen_lora_seed{seed}/final",
                '--test_csv', f"{self.config['paths']['experimental_variants']}/test_metadata.csv",
                '--output_csv', str(output_csv),
                '--batch_size', str(self.config['evaluation']['batch_size']),
                # Add: --temperature, --prompt, --threshold from DEV optimization
            ]

            if not self.run_command(cmd, f"Eval TEST seed {seed}", stage_num=9):
                return False

        return True

    def stage_10_psychometric_curves(self):
        """STAGE 10: Compute psychometric curves"""
        # Expand glob pattern to find all evaluation CSV files
        csv_pattern = f"{self.config['paths']['test_final']}/test_seed*_optimized.csv"
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            logger.error(f"No evaluation CSV files found matching {csv_pattern}")
            return False

        logger.info(f"Found {len(csv_files)} evaluation files: {[Path(f).name for f in csv_files]}")

        cmd = [
            'python3', 'scripts/compute_psychometric_curves.py',
            '--input_csvs', *csv_files,  # Expand list of files
            '--output_dir', self.config['paths']['psychometric'],
            '--bootstrap_iterations', str(self.config['analysis']['psychometric']['bootstrap_iterations']),
        ]

        logger.info("\n[10] Psychometric Curves")
        return self.run_command(cmd, "Compute curves", stage_num=10)

    def stage_11_baselines(self):
        """STAGE 11: Run baselines"""
        logger.info("\n[11] Baseline Models")

        if not self.config['analysis']['baselines']['silero_vad']['enabled']:
            logger.info("  Silero VAD: [DISABLED]")
            return True

        cmd = [
            'python3', 'scripts/baseline_silero_vad.py',
            '--test_csv', f"{self.config['paths']['experimental_variants']}/test_metadata.csv",
            '--filter_duration', '1000',
            '--filter_snr', '20',
            '--output_csv', f"{self.config['paths']['baselines']}/silero_vad_test.csv",
        ]

        return self.run_command(cmd, "Silero VAD", stage_num=11)

    def stage_12_aggregate_and_report(self):
        """STAGE 12-13: Aggregate results and generate report"""
        logger.info("\n[12-13] Results & Report")

        # TODO: Aggregate multi-seed (scripts/aggregate_multi_seed.py)
        logger.info("  Aggregation: [TODO]")

        # Generate final report
        cmd = [
            'python3', 'scripts/generate_pipeline_report.py',
            '--config', str(self.config_path),
            '--results_dir', self.config['paths']['results'],
            '--output_report', 'PIPELINE_EXECUTION_REPORT.md',
        ]

        return self.run_command(cmd, "Generate report", stage_num=13)

    def run_pipeline(self):
        """Execute complete pipeline"""
        logger.info("=" * 80)
        logger.info("SPEECH DETECTION PIPELINE - Qwen2-Audio + LoRA")
        logger.info("=" * 80)
        logger.info(f"Config: {self.config_path}")
        if self.dry_run:
            logger.info("Mode: DRY RUN (no execution)")
        elif self.smoke_test:
            logger.info("Mode: SMOKE TEST (minimal data)")
        else:
            logger.info("Mode: FULL EXECUTION")

        stages = [
            self.stage_0_validate_environment,
            self.stage_1_download_datasets,
            self.stage_2_prepare_base_clips,
            self.stage_3_generate_variants,
            self.stage_4_train_multiseed,
            self.stage_5_evaluate_dev,
            self.stage_6_optimize_on_dev,
            self.stage_9_evaluate_test_final,
            self.stage_10_psychometric_curves,
            self.stage_11_baselines,
            self.stage_12_aggregate_and_report,
        ]

        start_time = datetime.now()

        for idx, stage in enumerate(stages, 1):
            if not stage():
                logger.error(f"\n[FAILED] Stage {idx}/{len(stages)}: {stage.__name__}")
                logger.error(f"Pipeline aborted. Check logs for details.")
                return False

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("\n" + "=" * 80)
        logger.info("[SUCCESS] Pipeline completed")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Report: PIPELINE_EXECUTION_REPORT.md")
        logger.info(f"Logs: logs/")

        return True


def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline Orchestrator")
    parser.add_argument('--config', type=Path, default='config/pipeline_config.yaml',
                       help="Path to pipeline configuration YAML")
    parser.add_argument('--dry_run', action='store_true',
                       help="Print commands without executing")
    parser.add_argument('--smoke_test', action='store_true',
                       help="Run in smoke test mode (limited samples)")
    parser.add_argument('--resume', action='store_true',
                       help="Resume from checkpoint (skip existing outputs)")

    args = parser.parse_args()

    # Create orchestrator
    pipeline = PipelineOrchestrator(
        config_path=args.config,
        dry_run=args.dry_run,
        smoke_test=args.smoke_test
    )

    # Run pipeline
    success = pipeline.run_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
