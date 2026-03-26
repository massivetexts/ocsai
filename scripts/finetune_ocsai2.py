#!/usr/bin/env python3
"""CLI for submitting and monitoring OpenAI fine-tuning jobs for OCSAI 2.

Usage:
    # Submit format validation job (small)
    python scripts/finetune_ocsai2.py submit \
        --train data/training/ocsai2/cluster_split_train_sm.jsonl \
        --val data/training/ocsai2/cluster_split_val_sm.jsonl \
        --model gpt-4o-mini-2024-07-18 \
        --suffix ocsai2-sm

    # Submit full training job (large)
    python scripts/finetune_ocsai2.py submit \
        --train data/training/ocsai2/cluster_split_train_lg.jsonl \
        --val data/training/ocsai2/cluster_split_val_lg.jsonl \
        --model gpt-4o-mini-2024-07-18 \
        --suffix ocsai2-lg

    # Check status of most recent job
    python scripts/finetune_ocsai2.py status

    # Check status of specific job
    python scripts/finetune_ocsai2.py status ftjob-abc123

    # List recent fine-tuning jobs
    python scripts/finetune_ocsai2.py list
"""

import argparse
import sys
from pathlib import Path

import openai


def upload_or_load(client: openai.OpenAI, fname: Path) -> str:
    """Upload a file to OpenAI or return the ID of an already-uploaded file with the same name."""
    existing_files = [f for f in client.files.list() if f.filename == fname.name]
    if existing_files:
        print(f"Using already-uploaded file '{fname.name}' (id: {existing_files[0].id})")
        return existing_files[0].id
    else:
        print(f"Uploading {fname} ...")
        with open(fname, mode="rb") as f:
            result = client.files.create(file=f, purpose="fine-tune")
        print(f"  Uploaded as {result.id}")
        return result.id


def cmd_submit(args):
    client = openai.OpenAI()

    train_path = Path(args.train)
    val_path = Path(args.val) if args.val else None

    if not train_path.exists():
        print(f"Error: training file not found: {train_path}", file=sys.stderr)
        sys.exit(1)
    if val_path and not val_path.exists():
        print(f"Error: validation file not found: {val_path}", file=sys.stderr)
        sys.exit(1)

    # Suffix must be <= 18 chars
    if len(args.suffix) > 18:
        print(f"Error: suffix must be <= 18 characters (got {len(args.suffix)})", file=sys.stderr)
        sys.exit(1)

    # Upload files
    train_id = upload_or_load(client, train_path)
    val_id = upload_or_load(client, val_path) if val_path else None

    # Submit fine-tuning job
    print(f"\nSubmitting fine-tuning job:")
    print(f"  Base model: {args.model}")
    print(f"  Suffix:     {args.suffix}")
    print(f"  Train file: {train_path.name} ({train_id})")
    print(f"  Val file:   {val_path.name + ' (' + val_id + ')' if val_path else '(none)'}")

    create_kwargs = dict(
        training_file=train_id,
        model=args.model,
        suffix=args.suffix,
    )
    if val_id:
        create_kwargs["validation_file"] = val_id

    job = client.fine_tuning.jobs.create(**create_kwargs)

    print(f"\nJob submitted successfully!")
    print(f"  Job ID:     {job.id}")
    print(f"  Status:     {job.status}")
    print(f"\nMonitor with: python scripts/finetune_ocsai2.py status {job.id}")


def cmd_status(args):
    client = openai.OpenAI()

    if args.job_id:
        job = client.fine_tuning.jobs.retrieve(args.job_id)
    else:
        # Get most recent job
        jobs = list(client.fine_tuning.jobs.list(limit=1))
        if not jobs:
            print("No fine-tuning jobs found.")
            return
        job = jobs[0]

    print(f"Job ID:          {job.id}")
    print(f"Status:          {job.status}")
    print(f"Model:           {job.model}")
    print(f"Suffix:          {getattr(job, 'user_provided_suffix', 'N/A')}")
    print(f"Fine-tuned model:{' ' + job.fine_tuned_model if job.fine_tuned_model else ' (not yet available)'}")
    print(f"Hyperparameters: {job.hyperparameters}")
    if job.trained_tokens:
        print(f"Trained tokens:  {job.trained_tokens:,}")
    if job.error and job.error.message:
        print(f"Error:           {job.error.message}")

    # Show recent events
    print(f"\nRecent events:")
    events = list(client.fine_tuning.jobs.list_events(fine_tuning_job_id=job.id, limit=10))
    for event in reversed(events):
        print(f"  [{event.created_at}] {event.message}")


def cmd_list(args):
    client = openai.OpenAI()
    jobs = list(client.fine_tuning.jobs.list(limit=args.limit))

    if not jobs:
        print("No fine-tuning jobs found.")
        return

    print(f"{'Job ID':<35} {'Status':<15} {'Model':<30} {'Fine-tuned Model'}")
    print("-" * 120)
    for job in jobs:
        ft_model = job.fine_tuned_model or "(pending)"
        print(f"{job.id:<35} {job.status:<15} {job.model:<30} {ft_model}")


def main():
    parser = argparse.ArgumentParser(description="OCSAI 2 fine-tuning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # submit
    sub = subparsers.add_parser("submit", help="Upload files and submit a fine-tuning job")
    sub.add_argument("--train", required=True, help="Path to training JSONL file")
    sub.add_argument("--val", default=None, help="Path to validation JSONL file (optional)")
    sub.add_argument("--model", default="gpt-4o-mini-2024-07-18", help="Base model (default: gpt-4o-mini-2024-07-18)")
    sub.add_argument("--suffix", required=True, help="Model suffix (max 18 chars)")
    sub.set_defaults(func=cmd_submit)

    # status
    sub = subparsers.add_parser("status", help="Check fine-tuning job status")
    sub.add_argument("job_id", nargs="?", default=None, help="Job ID (defaults to most recent)")
    sub.set_defaults(func=cmd_status)

    # list
    sub = subparsers.add_parser("list", help="List recent fine-tuning jobs")
    sub.add_argument("--limit", type=int, default=10, help="Number of jobs to show (default: 10)")
    sub.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
