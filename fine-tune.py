import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from pathlib import Path
from tqdm import tqdm

from src.utils import get_logger, get_timestamp
from src.config import parse_args

import os

os.environ['NCCL_SOCKET_TIMEOUT'] = '7200'
os.environ['TORCH_DISTRIBUTED_DEFAULT_TIMEOUT'] = '7200'

args = parse_args()
accelerator = Accelerator()
args.local_rank = accelerator.process_index
logger = get_logger(__name__, args)

if accelerator.is_main_process:
    logger.info(f"Finetuning args: {json.dumps(vars(args), indent=4)}")

device = accelerator.device

model_name = args.model_name

if accelerator.is_main_process:
    logger.info(f"Loading model {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
    torch_dtype=torch.bfloat16,
    use_cache = False)
model.gradient_checkpointing_enable()

tokenizer.pad_token_id = 0
tokenizer.pad_token = "[pad]"
model.config.pad_token_id = tokenizer.pad_token_id

if accelerator.is_main_process:
    logger.info(f"Loading dataset {args.dataset_name}...")

dataset = load_dataset(args.dataset_name, "main")

def tokenize_function(examples):
    # no cot
    texts = [f"Only output the final result as just a single number without unit or explanation.\nQuestion: {x}\nAnswer: {y.strip().split("####")[-1].strip()}" for (x, y) in zip(examples['question'], examples['answer'])]
    tokenized = tokenizer(
        texts, 
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=args.max_data_length,
    )

    input_lens = [len(tokenizer(f"Only output the final result as just a single number without unit or explanation.\nQuestion: {x}\nAnswer: ").input_ids) for x in examples['question']]

    labels = tokenized.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    for i, input_len in enumerate(input_lens):
        labels[i, :input_len] = -100
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": labels
    }

dataset['train'] = dataset['train'].select(range(10))
split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=args.seed, shuffle=True)

tokenized_datasets = split_dataset.map(tokenize_function, batched=True, remove_columns=['question', 'answer'], load_from_cache_file=True, keep_in_memory=False)

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_dataloader  = torch.utils.data.DataLoader(eval_dataset,  batch_size=args.batch_size, shuffle=False)

if accelerator.is_main_process:
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
total_steps = (len(train_dataset) * args.num_epochs) // args.batch_size
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=args.warmup_ratio)

model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, scheduler
)

output_dir = Path(args.output_dir) / get_timestamp()
output_dir.mkdir(parents=True, exist_ok=True)

global_steps = 0

alpha = 0.01

def forward(model, batch):
    input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
    attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
    labels = torch.stack(batch['labels'], dim=1).to(device)

    hidden_states = model(
        input_ids, 
        attention_mask=attention_mask, 
        output_hidden_states=True,
    ).hidden_states[-1]

    original_hidden_states = hidden_states

    for t in range(args.num_loops - 1):
        hidden_states = model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
    logits = model.lm_head(original_hidden_states + alpha * hidden_states)
    # print(f"logits shape: {logits.shape}")
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

for epoch in range(args.num_epochs):
    model.train()
    pbar = tqdm(
        train_dataloader, 
        desc=f"Epoch {epoch + 1}/{args.num_epochs}", 
        disable=not accelerator.is_main_process
    )

    for batch in pbar:
        global_steps += 1
        optimizer.zero_grad()
        loss = forward(model, batch)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.local_rank == 0 and global_steps % args.logging_steps == 0:
            logger.info(f"Step {global_steps}: Loss = {loss.item()}")
    
    if (epoch + 1) % args.eval_steps == 0:
        if accelerator.is_main_process:
            logger.info(f"Evaluating model...")

        model.eval()

        losses = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            with torch.no_grad():
                loss = forward(model, batch)
                losses.append(loss.item())
        
        loss = torch.tensor(sum(losses) / len(losses), device=device)
        loss = accelerator.gather(loss).mean().item()

        if accelerator.is_main_process:
            logger.info(f"Test average perplexity: {loss}")

def save_model_checkpoint(accelerator, model, tokenizer, output_dir):
    accelerator.print("🔄 Waiting for all processes to sync before saving...")
    accelerator.wait_for_everyone()

    accelerator.print("📦 Gathering full state_dict for ZeRO-3 (if enabled)...")
    try:
        state_dict = accelerator.get_state_dict(model)
    except Exception as e:
        accelerator.print(f"❌ Failed to gather state_dict: {e}")
        return

    accelerator.print("💾 Saving model and tokenizer...")
    try:
        # unwrap_model 是必须的，否则 save_pretrained 可能不识别
        unwrapped_model = accelerator.unwrap_model(model)
        
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            accelerator.print(f"✅ Model and tokenizer saved to: {output_dir}")

            # 可选：验证保存是否成功
            try:
                from transformers import AutoModelForCausalLM
                _ = AutoModelForCausalLM.from_pretrained(output_dir)
                accelerator.print("🧪 Model reloaded successfully from saved directory.")
            except Exception as load_err:
                accelerator.print(f"❌ Reloading saved model failed: {load_err}")

    except Exception as save_err:
        accelerator.print(f"❌ Saving model failed: {save_err}")

save_model_checkpoint(
    accelerator=accelerator,
    model=model,
    tokenizer=tokenizer,
    output_dir=output_dir
)
accelerator.end_training()