import json
import torch
import re
from matplotlib import pyplot as plt
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

dataset = load_dataset(args.dataset_name, "default")

def extract_last_number(text):
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1]
    else:
        return None

def tokenize_function(examples):
    # no cot
    texts = [f"Only output the final result as just a single number without unit or explanation.\nQuestion: {x[0]["content"]}\nAnswer: {extract_last_number(x[1]["content"]).strip()}" for x in examples['messages']]

    tokenized = tokenizer(
        texts, 
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        max_length=args.max_data_length,
    )

    input_lens = [len(tokenizer(f"Only output the final result as just a single number without unit or explanation.\nQuestion: {x[0]["content"]}\nAnswer: ").input_ids) for x in examples['messages']]

    labels = tokenized.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    for i, input_len in enumerate(input_lens):
        labels[i, :input_len] = -100
    
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": labels
    }

dataset['train_119K'] = dataset['train_119K'].select(range(100))
del dataset['train']
split_dataset = dataset['train_119K'].train_test_split(test_size=0.1, seed=args.seed, shuffle=True)

tokenized_datasets = split_dataset.map(tokenize_function, batched=True, remove_columns=['id', 'messages'], load_from_cache_file=False, keep_in_memory=True)

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
    if args.num_loops > 1:
        logits = model.lm_head(original_hidden_states + alpha * hidden_states)
    else:
        logits = model.lm_head(original_hidden_states)
    # print(f"logits shape: {logits.shape}")
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss

def format_prompt(question: str) -> str:
    return (
        "Only output the final result as just a single number without unit or explanation.\n"
        f"Question: {question}\nAnswer: "
    )

dataset['train_119K'].select(range(100))

def eval_acc(model, tokenizer, accelerator):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(dataset['train_119K'], desc="Evaluating ACC", disable=not accelerator.is_main_process):
            example = example["messages"]
            question = example[0]["content"]
            gold_answer = extract_last_number(example[1]["content"])

            prompt = format_prompt(question)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            generated = input_ids.clone()
            
            for _ in range(10):
                attention_mask = torch.ones_like(generated)
                
                outputs = model(generated, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                original_hidden_states = hidden_states

                for _ in range(args.num_loops - 1):
                    hidden_states = model(inputs_embeds=hidden_states, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]

                logits = model.lm_head(original_hidden_states + alpha * hidden_states)

                next_token_logits = logits[:, -1, :]

                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                generated = torch.cat((generated, next_token.T), dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            if "####" in decoded:
                pred_answer = extract_last_number(decoded.split("####")[-1].strip().split("\n")[0])
            else:
                pred_answer = extract_last_number(decoded.split("Answer:")[-1].strip().split("\n")[0])
            if accelerator.is_main_process:
            #     print(decoded)
                print(pred_answer, gold_answer)
            total += 1
            if pred_answer is not None and abs(float(pred_answer) - float(gold_answer)) < 1e-7:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    return acc

history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
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

        if global_steps % args.logging_steps == 0:
            if accelerator.is_main_process:
                logger.info(f"Step {global_steps}: Loss = {loss.item()}")
                history['train_loss'].append(loss.item())
    
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

    acc = eval_acc(model=model, tokenizer=tokenizer, accelerator=accelerator)
    if accelerator.is_main_process:
        logger.info(f"Train average acc: {acc}")
        history['train_acc'].append(acc)

if accelerator.is_main_process:
    logger.info(f"Saving model checkpoint...")

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.legend()
plt.savefig('plot.png')

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
state_dict = accelerator.get_state_dict(model)
unwrapped_model.save_pretrained(
    output_dir,
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
    state_dict=state_dict,
)
if accelerator.is_main_process:
    tokenizer.save_pretrained(output_dir)

if accelerator.is_main_process:
    logger.info(f"Saved model to {output_dir}.")
accelerator.end_training()