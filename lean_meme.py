import os
import argparse
from functools import partial
import tqdm

import yaml

from scipy.stats import kendalltau, spearmanr
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
import accelerate



import data_meme
import meme_model


def save_checkpoint(model, output_dir, accelerator=None, optimizer=None, metrics=None, args=None, tokenizer=None):
    """
    Save the model and optimizer state to a checkpoint directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    if accelerator is None:
        model.save_to_directory(os.path.join(output_dir, 'model'))
    else:
        accelerator.unwrap_model(model).save_to_directory(os.path.join(output_dir, 'model'))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
    if args is not None:
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)
    if metrics is not None:
        with open(os.path.join(output_dir, 'metrics.yaml'), 'w') as f:
            yaml.dump(metrics, f)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))


def lr_lambda_linear_with_min_lr(step, args):
    peak_lr = args.peak_lr
    min_lr = peak_lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return step / args.warmup_steps
    slope = (peak_lr - min_lr) / (args.max_steps - args.warmup_steps)
    intercept = peak_lr - slope * args.warmup_steps
    current_lr = - slope * (step - args.warmup_steps) + intercept
    return max(current_lr, min_lr) / peak_lr


class FakeWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass

def main():
    parser = argparse.ArgumentParser(description="Train a model on the MMAU dataset.")
    parser.add_argument('--train_data', type=str, nargs='+',
                        default=['/mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_human.json',
                                 '/mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/train_llmj_p1.json',],
                        help='Paths to training data files. Can be multiple files for different splits.')
    parser.add_argument('--val_data', type=str,
                        default='/mnt/matylda4/kesiraju/tools/potato/data_splits_for_meme/M1_6_M2_5/val_human.json',
                        help='Paths to human data files for training.')
    parser.add_argument('--max_data_length', type=int, default=1000,
                        help="Maximum length of input text in characters. Avoids OOM errors for some rather loquacious "
                             "models/judges.")
    parser.add_argument('--dataset_sampling_weights', type=float, nargs='+', default=None,
                        help='Weights for sampling from different datasets. '
                             'If not provided, datasets will just be concatenated.')
    parser.add_argument('--skip_rationale', action='store_true',
                        help='If set, the model will not use rationales for scoring. '
                             'This may be useful for training on datasets without rationales.')
    parser.add_argument('--skip_question', action='store_true',
                        help='If set, the model will not use questions for scoring. '
                             'This is mostly for analysis.')
    parser.add_argument('--add_transcript', action='store_true',
                        help='If set, the model will use transcriptions for scoring.')

    parser.add_argument('--prompts_yaml', type=str, default=None,
                        help='Path to YAML file with prompts for the model. If not provided, no prompts will be used.')
    parser.add_argument('--score_type', type=str, default='bernoulli',
                        choices=['bernoulli', 'beta', 'mse', 'bmm'],
                        help='Type of scoring to use for the model. Options are "bernoulli" or "beta".')
    parser.add_argument('--model', type=str, default='google/gemma-3-1b-it',
                        help='LLM to initialize from.')
    parser.add_argument('--tokenizer', type=str,
                        help="Path to the tokenizer directory. Defaults to model if unset")
    parser.add_argument('--layers_to_use', nargs='+', type=int, default=None,
                        help='List of layer indices to use for scoring. If not provided, all layers will be used. ')

    parser.add_argument('--lora_rank', type=int,
                               help='LoRA rank, default is no LoRA')
    parser.add_argument('--quantization_level', type=str, default='none',
                        choices=['none', '4bit', '8bit'],
                        help='Quantization level to use for the model. Default is no quantization.')
    parser.add_argument('--use_cls_token', action='store_true',
                        help='If set, the model will append a CLS token for scoring.')

    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to a checkpoint to load the model from. If provided, the model will be loaded from this checkpoint ')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume training from the latest checkpoint.')

    parser.add_argument('--max_steps', type=int, default=4000, help='Number of epochs to train.')
    parser.add_argument('--val_steps', type=int, default=100, help='Number of steps between validation.')
    parser.add_argument('--save_steps', type=int, default=500, help='Number of steps between saves.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps. Useful for large models with small batch sizes.')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps for learning rate.')
    parser.add_argument('--peak_lr', type=float, default=1e-5, help='Peak learning rate for training.')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                        help='Minimum learning rate ratio relative to peak learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,)

    parser.add_argument('--early_stopping_patience', type=int, default=30,
                        help='Number of validation steps without improvement before early stopping.')
    parser.add_argument('--lr_ratio_classifier', type=float, default=1.0,
                        help='Learning rate ratio for the classifier head relative to the LM.')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs for TensorBoard.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the trained model.')

    args = parser.parse_args()

    ddp_kwarg_handler = accelerate.utils.DistributedDataParallelKwargs(
        find_unused_parameters=True
    )
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwarg_handler])
    torch.tensor(0).to(accelerator.device)
    device = accelerator.device

    for arg in vars(args):
        accelerator.print(f'{arg}: {getattr(args, arg)}')


    # Load the model and tokenizer
    accelerator.print(f'Loading model {args.model} with {args.quantization_level if args.quantization_level != "none" else "bfloat16"} precision and flash attention 2')
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.quantization_level == '4bit',
        load_in_8bit= args.quantization_level == '8bit',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    if args.quantization_level == 'none':
        lm = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation='flash_attention_2',
            device_map=device.type,
            low_cpu_mem_usage=True,
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(args.model,
                                                  # torch_dtype=torch.bfloat16,
                                                  attn_implementation='flash_attention_2',
                                                  quantization_config=bnb_config,
                                                  device_map=device.type,
                                                  low_cpu_mem_usage=True,
                                                  )

    # if hasattr(lm, 'model'):
    #     # Remove the classification head if it exists
    #     lm = lm.model
    if args.tokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    from peft import LoraConfig, get_peft_model, PeftModel

    if args.lora_rank is not None and not args.load_checkpoint:
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            target_modules='all-linear',
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
        )
        lm = get_peft_model(lm, lora_config)

    scoring_model = meme_model.Meme(
        lm, score_type=args.score_type,
        layers_to_use=args.layers_to_use,
        use_cls_token=args.use_cls_token,
        ).to(torch.bfloat16)


    if args.prompts_yaml is not None:
        prompts = yaml.safe_load(open(args.prompts_yaml, 'r'))
    else:
        prompts = None

    collate_fn = data_meme.CollateFn(tokenizer)
    if args.dataset_sampling_weights is None:
        train_dataset = data_meme.ConcatenatedDataset(
            [data_meme.UnifiedAnnotationDataset(
                jfile,
                prompts=prompts,
                filter_func=lambda x: len(x['text']) <= args.max_data_length,
                skip_rationale=args.skip_rationale,
                skip_question=args.skip_question,
                add_transcript=args.add_transcript,
            ) for jfile in args.train_data]
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_fn
        )
    else:

        train_datasets = (
            [data_meme.UnifiedAnnotationDataset(
                jfile,
                prompts=prompts,
                filter_func=lambda x: len(x['text']) <= args.max_data_length,
                skip_rationale=args.skip_rationale,
                skip_question=args.skip_question,
                add_transcript=args.add_transcript,
            ) for jfile in args.train_data]
        )

        train_dataloaders = [DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn
        ) for train_dataset in train_datasets]
        train_dataset = data_meme.DatasetWithSampling(train_dataloaders,
                                                      sampling_weights=args.dataset_sampling_weights)
        train_dataloader = DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, collate_fn=lambda x: x[0],
        )

    val_dataset = data_meme.UnifiedAnnotationDataset(
        args.val_data,
        prompts=prompts,
        skip_rationale=args.skip_rationale,
        skip_question=args.skip_question,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    start_step = 0
    best_val_rho = float('-inf')
    all_metrics = []
    metrics_at_best_checkpoint = []
    train_loss = 0
    train_count = 0

    if args.load_checkpoint:
        accelerator.print(f'Loading model from checkpoint: {args.load_checkpoint}')
        if args.lora_rank is not None:
            lm = PeftModel.from_pretrained(lm, os.path.join(args.load_checkpoint, 'lm'))
            accelerator.print('Loaded LoRA model from checkpoint')
        else:
            lm = AutoModel.from_pretrained(
                os.path.join(args.load_checkpoint, 'lm'),
                torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2'
            )
            accelerator.print('Loaded LM model from checkpoint')
        scoring_model = meme_model.Meme.load_from_directory(
            args.load_checkpoint, lm, device=device
        )
    if args.resume:
        accelerator.print('Resuming from latest checkpoint')
        latest_checkpoint = os.path.join(args.output_dir, 'latest')
        if args.lora_rank is not None:
            lm = PeftModel.from_pretrained(lm, os.path.join(latest_checkpoint, 'lm'))
            accelerator.print('Loaded LoRA model from checkpoint')
        else:
            lm = AutoModel.from_pretrained(
                os.path.join(latest_checkpoint, 'lm'),
                torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2'
            )
            accelerator.print('Loaded LM model from checkpoint')
        scoring_model = meme_model.Meme.load_from_directory(
            latest_checkpoint, lm, device=device
        )
        with open(os.path.join(latest_checkpoint, 'metrics.yaml'), 'r') as f:
            all_metrics = yaml.safe_load(f)
        start_step = all_metrics[-1]['step'] if all_metrics else 0
        best_val_rho = all_metrics[-1].get('mean_rho', best_val_rho) if all_metrics else float('-inf')

    optimizer_dict_list = []
    optimizer_dict_list.append({'params': [p for p in scoring_model.lm.parameters() if p.requires_grad],
                                'lr': args.peak_lr})
    optimizer_dict_list.append(
        {'params': scoring_model.linear.parameters(), 'lr': args.peak_lr * args.lr_ratio_classifier}
    )
    optimizer = torch.optim.AdamW(optimizer_dict_list, lr=args.peak_lr, weight_decay=args.weight_decay)
    if args.resume:
        accelerator.print('Resuming optimizer state from checkpoint')
        latest_checkpoint = os.path.join(args.output_dir, 'latest')
        optimizer.load_state_dict(torch.load(os.path.join(latest_checkpoint, 'optimizer.pt')))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=partial(lr_lambda_linear_with_min_lr, args=args)
    )
    scoring_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        scoring_model, optimizer, train_dataloader, val_dataloader
    )
    accelerator.print(f'Number of total parameters: {sum(p.numel() for p in scoring_model.parameters())}')
    accelerator.print(
        f'Number of trainable parameters: {sum(p.numel() for p in scoring_model.parameters() if p.requires_grad)}')
    accelerator.print(
        f'Total batch size: {args.batch_size * accelerator.num_processes * args.accumulation_steps} ',
        f'({accelerator.num_processes} processes, ',
        f'per process: {args.batch_size}, accumulation steps: {args.accumulation_steps})'
    )
    accelerator.print(
        f'Total number of training samples: {len(train_dataset)}'
    )
    accelerator.print(
        f'Total number of validation samples: {len(val_dataset)}'
    )

    if accelerator.is_main_process and args.log_dir:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        accelerator.print('Not using TensorBoard logging, as log_dir is not set.')
        writer = FakeWriter()

    train_loader_iter = iter(train_dataloader)
    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = {arg: getattr(args, arg) for arg in vars(args)}
    if not os.path.exists(os.path.join(args.output_dir, 'args.yaml')):
        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(args_dict, f)
    for j in tqdm.tqdm(range(start_step + 1, args.max_steps + 1), disable=not accelerator.is_main_process):
        for k in range(args.accumulation_steps):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_dataloader)
                batch = next(train_loader_iter)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = scoring_model(batch)  # ,
                                        # prior_params=torch.tensor([[0., 0.]]).to(device),)
                # outputs = torch.utils.checkpoint.checkpoint(
                #     scoring_model, batch, use_reentrant=False,
                # )
                loss = outputs['loss']
            accelerator.backward(loss/args.accumulation_steps)
            loss = accelerator.gather(loss.detach()).mean().item()  # Average loss across all processes
            train_loss += (loss / args.accumulation_steps)
        train_count += 1
        grad_norm = accelerator.clip_grad_norm_(scoring_model.parameters(), args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process:
            writer.add_scalar("Loss/train_loss", loss, j)
            writer.add_scalar(f"Loss/train_loss_{args.score_type}", loss, j)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], j)
            writer.add_scalar("Data/Text Length", batch['input_len'].float().mean().item(), j)
            writer.add_scalar("Data/Batch Size", batch['input_ids'].shape[0], j)
            writer.add_scalar("Loss/Grad Norm", grad_norm, j)

        if j % args.val_steps == 0:
            val_loss = 0
            val_count = 0
            all_val_annotations = []
            all_val_predictions = []
            all_val_params = []

            all_val_variance_annotations = []
            all_val_variance_predictions = []

            with torch.no_grad():
                for val_batch in tqdm.tqdm(val_dataloader, disable=not accelerator.is_main_process):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        val_outputs = scoring_model(val_batch)
                        loss = val_outputs['loss']
                    loss = accelerator.gather(loss).mean()
                    val_loss += loss.item()
                    val_count += 1

                    val_annotation = val_batch['average_labels']
                    val_prediction = val_outputs['prob1']
                    all_val_annotations.extend(val_annotation.cpu().tolist())
                    all_val_predictions.extend(val_prediction.cpu().tolist())
                    all_val_params.extend(val_outputs['params'].cpu().tolist())

                    if 'label_variance' in val_batch:
                        val_variance_annotation = val_batch['label_variance']
                        val_variance_prediction = val_outputs['variance']
                        all_val_variance_annotations.extend(val_variance_annotation.cpu().tolist())
                        all_val_variance_predictions.extend(val_variance_prediction.cpu().tolist())
                    metrics_to_sync = torch.tensor([val_loss, val_count], device=device)
                    metrics_to_sync = accelerator.gather(metrics_to_sync)
                    val_loss = metrics_to_sync[0].sum().item()
                    val_count = metrics_to_sync[1].sum().item()

                with open(os.path.join(args.output_dir, f'val_predictions_{j}_{accelerator.process_index}.yaml'), 'w') as f:
                    yaml.dump({
                        'predictions': all_val_predictions,
                        'annotations': all_val_annotations,
                        'params': all_val_params,
                        'predictions_variance': all_val_variance_predictions,
                        'annotations_variance': all_val_variance_annotations,
                    }, f)
                    accelerator.wait_for_everyone()

            all_val_annotations = []
            all_val_predictions = []
            all_val_params = []
            all_val_variance_annotations = []
            all_val_variance_predictions = []
            for i in range(accelerator.num_processes):
                with open(os.path.join(args.output_dir, f'val_predictions_{j}_{i}.yaml'), 'r') as f:
                    data = yaml.safe_load(f)
                    all_val_annotations.extend(data['annotations'])
                    all_val_predictions.extend(data['predictions'])
                    all_val_params.extend(data['params'])
                    all_val_variance_annotations.extend(data['annotations_variance'])
                    all_val_variance_predictions.extend(data['predictions_variance'])


            # Calculate Kendall's tau and Spearman's rank correlation
            kendall_tau = kendalltau(all_val_annotations, all_val_predictions)
            spearman_corr = spearmanr(all_val_annotations, all_val_predictions)

            variance_kendall_tau = kendalltau(all_val_variance_annotations, all_val_variance_predictions)
            variance_spearman_corr = spearmanr(all_val_variance_annotations, all_val_variance_predictions)

            if accelerator.is_main_process:
                writer.add_scalar("Loss/val_loss_mean", val_loss / val_count, j)
                writer.add_scalar(f"Loss/val_loss_mean_{args.score_type}", val_loss / val_count, j)
                writer.add_scalar("Correlations/Mean Kendall's Tau", float(kendall_tau.statistic), j)
                writer.add_scalar("Correlations/Mean Spearman's Rank Correlation", float(spearman_corr.statistic),
                                  j)
                writer.add_scalar("Correlations/Variance Kendall's Tau", float(variance_kendall_tau.statistic), j)
                writer.add_scalar("Correlations/Variance Spearman's Rank Correlation",
                                  float(variance_spearman_corr.statistic), j)

            logging_dict = {
                'step': j,
                'train_loss': train_loss / train_count,
                'val_loss': val_loss / val_count,
                'best_val_rho': best_val_rho if best_val_rho > float(spearman_corr.statistic) else float(spearman_corr.statistic),
                'variance_tau': float(variance_kendall_tau.statistic),
                'variance_rho': float(variance_spearman_corr.statistic),
                'mean_tau': float(kendall_tau.statistic),
                'mean_rho': float(spearman_corr.statistic),
            }
            accelerator.print(logging_dict)
            all_metrics.append(logging_dict)
            if float(spearman_corr.statistic) > best_val_rho:
                best_val_rho = float(spearman_corr.statistic)
                save_checkpoint(scoring_model, os.path.join(args.output_dir, 'best'),
                                accelerator=accelerator, metrics=all_metrics, args=args,
                                tokenizer=tokenizer)
                metrics_at_best_checkpoint = logging_dict
            if accelerator.is_main_process:
                save_checkpoint(scoring_model, os.path.join(args.output_dir, 'latest'),
                                accelerator=accelerator, metrics=all_metrics, args=args, optimizer=optimizer,
                                tokenizer=tokenizer)

        if accelerator.is_main_process and j % args.save_steps == 0:
            save_checkpoint(scoring_model, os.path.join(args.output_dir, f'checkpoint_{j}'),
                            accelerator=accelerator, metrics=all_metrics, args=args, tokenizer=tokenizer)

        recent_val_rho = [m['mean_rho'] for m in all_metrics[-args.early_stopping_patience:]]
        all_val_rhos = [m['mean_rho'] for m in all_metrics]
        if len(all_val_rhos) > args.early_stopping_patience and all(
                v < max(all_val_rhos) for v in recent_val_rho):
            accelerator.print('Early stopping')
            break

    accelerator.print('Training finished. Metrics at best step:')
    accelerator.print(metrics_at_best_checkpoint)
if __name__ == '__main__':
    main()
