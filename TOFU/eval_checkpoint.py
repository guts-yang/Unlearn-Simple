"""Evaluate an already-trained TOFU unlearning checkpoint and write aggregate_stat.txt."""
from data_module import TextForgetDatasetQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
import transformers
import os
from pathlib import Path
from utils import get_model_identifiers_from_yaml, set_random_seed


@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    seed = cfg.seed
    set_random_seed(seed)

    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    else:
        local_rank = 0
        device_map = None

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    unlearned_path = cfg.get('unlearned_path', None)
    if unlearned_path is None:
        raise ValueError("Pass unlearned_path=/path/to/checkpoint-XX")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print("######################")
    print("Evaluating unlearned model:", unlearned_path)
    print("Oracle / base model:", cfg.model_path)
    print("Saving to:", cfg.save_dir)
    print("######################")

    max_length = 500
    if cfg.split in ['forget01', 'forget05', 'forget10']:
        data_path = 'locuslab/TOFU'
    else:
        raise NotImplementedError

    torch_format_dataset = TextForgetDatasetQA(
        data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=max_length,
        split=cfg.split,
        loss_type=cfg.forget_loss,
    )

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0,
        max_steps=1,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=10,
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="no",
        ddp_find_unused_parameters=False,
        deepspeed='config/ds_config.json',
        weight_decay=cfg.weight_decay,
        evaluation_strategy="no",
        report_to=[],
    )

    print("Loading unlearned checkpoint")
    model = AutoModelForCausalLM.from_pretrained(
        unlearned_path,
        use_flash_attention_2=model_cfg["flash_attention2"] == "true",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print("Loading oracle model")
    oracle_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        use_flash_attention_2=model_cfg["flash_attention2"] == "true",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    if cfg.split in ['forget01', 'forget05', 'forget10']:
        pass
    else:
        raise NotImplementedError

    # Match forget.yaml eval.save_dir to the run directory that already holds checkpoint-55.
    cfg.eval.save_dir = cfg.save_dir
    cfg.eval.overwrite = True

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator_forget,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
        seed=seed,
        ref_policy=cfg.ref_policy,
        beta=cfg.beta,
        gamma=cfg.gamma,
        npo_coeff=cfg.npo_coeff,
        grad_diff_coeff=cfg.grad_diff_coeff,
        KL_coeff=cfg.KL_coeff,
    )
    model.config.use_cache = False

    # Write metrics into checkpoint-55 (or whatever step is passed).
    step = int(cfg.get('eval_step', 55))
    trainer.state.global_step = step
    trainer.evaluate()
    print(f"Done. Check {cfg.save_dir}/checkpoint-{step}/aggregate_stat.txt")


if __name__ == "__main__":
    main()
