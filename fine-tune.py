import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from pathlib import Path
from tqdm import tqdm

from src.utils import get_logger, get_timestamp
from src.config import parse_args

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
    torch_dtype=torch.bfloat16)
model.gradient_checkpointing_disable()

tokenizer.pad_token_id = 0
tokenizer.pad_token = "[pad]"
model.config.pad_token_id = tokenizer.pad_token_id

if accelerator.is_main_process:
    logger.info(f"Loading dataset {args.dataset_name}...")

dataset = load_dataset(args.dataset_name, "main")

def tokenize_function(examples):
    texts = [f"Question: {x}\nAnswer:{y}" for (x, y) in zip(examples['question'], examples['answer'])]

    tokenized = tokenizer(
        texts, 
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=args.max_data_length,
    )

    input_lens = [len(tokenizer(f"Question: {x}\nAnswer:").input_ids) for x in examples['question']]

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

tokenized_datasets = split_dataset.map(tokenize_function, batched=True, remove_columns=['question', 'answer'])

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

def forward(model, batch):
    input_ids = torch.stack(batch['input_ids'], dim=1).to(device)
    attention_mask = torch.stack(batch['attention_mask'], dim=1).to(device)
    labels = torch.stack(batch['labels'], dim=1).to(device)

    print(f"input_ids.shape: {input_ids.shape}")
    print(f"attention_mask.shape: {attention_mask.shape}")
    print(f"labels.shape: {labels.shape}")

    outputs = model(
        input_ids, 
        attention_mask=attention_mask,
        labels=labels,
    )

    print(f"logits.shape: {outputs.logits.shape}")

    return outputs

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
        outputs = forward(model, batch)
        loss = outputs.loss
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.local_rank == 0 and global_steps % args.logging_steps == 0:
            logger.info(f"Step {global_steps}: Loss = {loss.item()}")
    
    if (epoch + 1) % args.save_steps == 0 and accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        logger.info(f"Saving model checkpoint...")
        torch.save(unwrapped_model.state_dict(), output_dir / f"checkpoint-{epoch + 1}.pt")
        logger.info(f"Model checkpoint saved to {output_dir / f'checkpoint-{epoch + 1}.pt'}")
    
    if (epoch + 1) % args.eval_steps == 0:
        if accelerator.is_main_process:
            logger.info(f"Evaluating model...")

        model.eval()

        losses = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            with torch.no_grad():
                outputs = forward(model, batch)
                losses.append(outputs.loss.item())
        
        loss = torch.tensor(sum(losses) / len(losses), device=device)
        loss = accelerator.gather(loss).mean().item()

        if accelerator.is_main_process:
            logger.info(f"Test average perplexity: {loss}")
