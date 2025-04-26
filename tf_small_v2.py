import os
import subprocess
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_scheduler
)
from torch.optim import AdamW
from datasets import load_dataset
from evaluate import load as load_metric
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Auto‑install metric dependencies if needed
# ────────────────────────────────────────────────────────────────────────────────
def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

try:
    import evaluate
    import rouge_score
    import sacrebleu
    import nltk
except ImportError:
    _pip_install([
        "evaluate",
        "rouge-score",
        "absl-py",
        "nltk",
        "sacrebleu",
        "sentencepiece"
    ])
    import evaluate

nltk.download("punkt", quiet=True)
# Disable wandb logging from transformers
os.environ["WANDB_DISABLED"] = "true"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "10.10.1.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)

def cleanup():
    dist.destroy_process_group()

# NEW: Method to save the trained model after training
def save_trained_model(model, checkpoint_dir):
    """
    SAVES THE TRAINED MODEL TO THE SPECIFIED DIRECTORY.
    """
    wrapped = model.module if isinstance(model, DDP) else model
    wrapped.save_pretrained(checkpoint_dir)
    print(f"MODEL SUCCESSFULLY SAVED TO {checkpoint_dir}")


def main(rank, world_size, eval_only, checkpoint_dir):
    assert torch.cuda.is_available(), f"CUDA not available on rank {rank}"
    device = torch.device("cuda", 0)
    print(f"[Rank {rank}] Using GPU: {torch.cuda.get_device_name(device)}")

    setup(rank, world_size)

    # 1) LOAD & SUBSAMPLE FULL_DATA CONFIG
    dataset_full = load_dataset("ServiceNow-AI/M2Lingual", "full_data")  # CHANGED CONFIG
    subset = dataset_full["train"]                                        # TAKE TRAIN SPLIT
    subset = subset.shuffle(seed=42).select(range(13_000))                # TRUNCATION STEP
    split = subset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # 2) TOKENIZER & PREPROCESSING
    model_name = "t5-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def preprocess(ex):
        # ADD LANGUAGE AND TASK TO THE PROMPT
        header = f"LANGUAGE: {ex['language']} | TASK: {ex['task']}"
        # INCLUDE SEED AND EVOLVED USER PROMPTS
        header += f" | SEED_PROMPT: {ex.get('seed_prompt','')} | EVOLVED_PROMPT: {ex.get('evolved_user_prompt','')}"
        # FLATTEN THE FULL CONVERSATION HISTORY
        conv_history = " ".join([turn.get("content","") for turn in ex["conversation"]])
        input_text = " ".join([header, conv_history])

        # EXPLICITLY SPECIFY TEXT FOR SOURCE
        inputs = tokenizer(
            text=input_text,
            padding="max_length",
            truncation=True,
            max_length=512
        )

        # SET TARGET WITH text_target KEYWORD
        target_text = ex.get("output_assistant_reply", "")
        targets = tokenizer(
            text_target=target_text,
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs["labels"] = targets.input_ids
        return inputs

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_tok  = eval_ds.map(preprocess,  remove_columns=eval_ds.column_names)
    cols = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=cols)
    eval_tok .set_format(type="torch", columns=cols)

    # 3) DATALOADERS
    train_loader = train_sampler = None
    if not eval_only:
        train_sampler = DistributedSampler(train_tok, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader  = DataLoader(train_tok, sampler=train_sampler, batch_size=4)
    eval_loader = DataLoader(eval_tok, batch_size=8)

    # 4) LOAD OR RESUME MODEL
    src = checkpoint_dir if eval_only else model_name
    model = AutoModelForSeq2SeqLM.from_pretrained(src).to(device)
    model = DDP(model, device_ids=[0])

    # 5) TRAINING LOOP
    if not eval_only:
        optimizer    = AdamW(model.parameters(), lr=5e-5)
        num_steps    = len(train_loader) * 3
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_steps
        )
        model.train()
        for epoch in range(3):
            train_sampler.set_epoch(epoch)
            if rank == 0:
                print(f"\n▶️  Starting epoch {epoch}")
            for step, batch in enumerate(tqdm(train_loader, desc=f"R{rank}-E{epoch}")):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if rank == 0 and step % 200 == 0:
                    print(f"  [epoch {epoch} | step {step}/{len(train_loader)}] loss = {loss.item():.4f}")
            if rank == 0:
                print(f"✅  Finished epoch {epoch}; last loss = {loss.item():.4f}")
        # SAVE AFTER TRAINING
        if rank == 0:
            save_trained_model(model, checkpoint_dir)  # NEW: SAVE MODEL ONCE

    # 6) EVALUATION
    if rank == 0:
        model.eval()
        rouge = load_metric("rouge")
        bleu  = load_metric("bleu")
        all_preds, all_refs = [], []
        for batch in tqdm(eval_loader, desc="Generating"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gen_ids = model.module.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            refs  = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            for p, r in zip(preds, refs):
                if r.strip():
                    all_preds.append(p)
                    all_refs.append(r)
        if not all_refs:
            print("No non-empty references found; skipping BLEU/ROUGE.")
        else:
            bleu_scores  = bleu.compute(predictions=all_preds, references=all_refs)
            rouge_scores = rouge.compute(predictions=all_preds, references=all_refs, use_stemmer=True)
            print("\n=== Generation Metrics ===")
            print(f"BLEU: {bleu_scores['bleu']:.4f}")
            for name, score in rouge_scores.items():
                 print(f"ROUGE-{name.upper()}: {score:.4f}")
        model.module.save_pretrained("./outputs_gen/")

    cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",       type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--eval_only",  action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs_gen")
    args = parser.parse_args()
    main(args.rank, args.world_size, args.eval_only, args.checkpoint_dir)
