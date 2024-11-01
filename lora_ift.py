import argparse
import os
from tqdm import tqdm

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
import wandb

from utils import seed_everything, CustomDataset


def train_model(args, model, train_dl, epochs, optimizer, log_every_n_steps=50):
    model.train()
    print("Start training...")
    
    # logging
    global_step = 0
    running_loss = 0.0
    steps_since_last_log = 0
    
    for epoch in range(epochs):
        if accelerator.is_main_process:
            train_iter = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            train_iter = train_dl
            
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_iter):
            with accelerator.accumulate(model):
                batch['labels'] = batch['input_ids'].clone()
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                
                optimizer.step()
                optimizer.zero_grad()

                # update
                loss_value = loss.item()
                running_loss += loss_value
                epoch_loss += loss_value
                steps_since_last_log += 1
                
            # log every n steps
            if accelerator.is_main_process and global_step % log_every_n_steps == 0:
                avg_running_loss = running_loss / steps_since_last_log
                wandb.log({
                    "train/loss": avg_running_loss,
                    "train/global_step": global_step,
                })
                
                train_iter.set_postfix({
                    'loss': f'{avg_running_loss:.4f}',
                    'step': global_step
                })
                
                running_loss = 0.0
                steps_since_last_log = 0
            
            global_step += 1
            
        # each epoch
        avg_epoch_loss = epoch_loss / len(train_dl)
        if accelerator.is_main_process:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch + 1
            })
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                os.path.join(args.weights, args.exp_name+f"_{epoch+1}"), 
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )


def print_trainable_parameters(model):
    if accelerator.is_main_process:
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--train_file", type=str, default="data/benjamin_harrison/train/Benjamin_Harrison_speeches_instruction_dataset_100%.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1006)
    parser.add_argument("--weights", type=str, default="weights/")
    args = parser.parse_args()

    # exp name
    name = args.train_file.split("/")[1]
    percentage = args.train_file.split("/")[-1].split("_")[-1][:-4]
    args.exp_name = f"{name}_{percentage}"

    # accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.device = accelerator.device
    print(f"Device: {args.device}")

    # seed
    seed_everything(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # lora config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type='CAUSAL_LM'
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map=args.device,
        use_cache=False
    )
    model.gradient_checkpointing_enable()  # enable gradient checkpointing to save memory
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    # data
    train_ds = CustomDataset(args.train_file, tokenizer)
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # prepare through accelerator
    model, optimizer, train_dl = accelerator.prepare(
        model, optimizer, train_dl
    )

    # wandb
    if accelerator.is_main_process:
        wandb.init(project="style-instruct", name=args.exp_name, entity="irisiris")
        wandb.config.update(args)

    # training loop
    train_model(args, model, train_dl, epochs=args.epochs, optimizer=optimizer)

    # done
    wandb.finish()
    