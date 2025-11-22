import json
import os
import argparse
import tqdm

import yaml
from peft import PeftModel

from scipy.stats import kendalltau, spearmanr
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
import accelerate



import data_meme
import meme_model


def main():
    parser = argparse.ArgumentParser(description="Meme Inference Script")
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for DataLoader')

    parser.add_argument('--tokenizer_path', type=str,
                        help="Path to the tokenizer directory. "
                             "This is there for backward compatibility as I wasn't saving the tokenizer earlier on.")
    parser.add_argument('--test_set_is_labeled', action='store_true',
                        help='If set, the test set is assumed to be labeled. '
                             'This is used to decide whether to compute Kendall Tau and Spearman correlation scores.')
    parser.add_argument('--skip_rationale', action='store_true',
                        help='If set, the model will not use rationales for scoring. '
                             'This may be useful for training on datasets without rationales.')
    parser.add_argument('--skip_question', action='store_true',
                        help='If set, the model will not use questions for scoring. '
                             'This is mostly for analysis.')
    parser.add_argument('--add_transcript', action='store_true',
                        help='If set, the model will use transcript as additional context.')

    parser.add_argument('--meme_model_path', type=str,
                        default='/mnt/matylda5/iyusuf/exps/better_score/models//lean_meme-context-beta-train_human-val_M1_6_M2_5-model_gemma12bit_lora256/best/model',
                        help='Path to the trained Meme model directory')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the json file containing the data to score.')
    parser.add_argument('--output_dir', type=str, default='meme_inference_output',
                        help='Directory to save inference results')

    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    torch.tensor(0).to(accelerator.device)

    for arg in vars(args):
        accelerator.print(f'{arg}: {getattr(args, arg)}')

    os.makedirs(args.output_dir, exist_ok=True)

    if args.test_set_is_labeled:
        inference_data = data_meme.UnifiedAnnotationDataset(args.data,
                                                            skip_question=args.skip_question,
                                                            skip_rationale=args.skip_rationale,
                                                            add_transcript=args.add_transcript,
                                                            )
    else:
        inference_data = data_meme.InferenceDataset(args.data,
                                                    skip_question=args.skip_question,
                                                    skip_rationale=args.skip_rationale,
                                                    add_transcript=args.add_transcript,
                                                    )

    # Load the Meme model
    if os.path.isfile(os.path.join(args.meme_model_path, 'lm', 'adapter_config.json')):
        base_model_name = json.load(open(os.path.join(args.meme_model_path, 'lm', 'adapter_config.json'))
                                    )['base_model_name_or_path']
        lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device.type,
            low_cpu_mem_usage=True,
        )
        lm = PeftModel.from_pretrained(
            lm,
            os.path.join(args.meme_model_path, 'lm',),
            low_cpu_mem_usage=True,
        )
        lm = lm.merge_and_unload()
    else:
        lm = AutoModel.from_pretrained(
            os.path.join(args.meme_model_path, 'lm'),
            device_map=device.type,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(os.path.dirname(args.meme_model_path), 'tokenizer'),
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    scoring_model = meme_model.Meme.load_from_directory(args.meme_model_path, lm=lm, device=device)

    collate_fn = data_meme.CollateFn(tokenizer)
    inference_loader = DataLoader(
        inference_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=2,
    )

    scoring_model.eval()
    scoring_model, inference_loader = accelerator.prepare(scoring_model, inference_loader)
    with torch.inference_mode():
        all_idxs = []
        all_params = []
        all_prob1 = []
        all_variance = []
        all_ratings = []  # Only used if test_set_is_labeled is True
        all_ground_truth_variances = []  # Only used if test_set_is_labeled is True
        for batch in tqdm.tqdm(inference_loader, desc="Inference"):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                outputs = scoring_model(batch)
            params = outputs['params'].cpu().tolist()
            prob1 = outputs['prob1'].cpu().tolist()
            variance = outputs['variance'].cpu().tolist()

            all_idxs.extend(batch['idx'])
            all_params.extend(params)
            all_prob1.extend(prob1)
            all_variance.extend(variance)
            if args.test_set_is_labeled:
                all_ratings.extend(batch['average_labels'].cpu().tolist())
                all_ground_truth_variances.extend(batch['label_variance'].cpu().tolist())

        # Save results
        result = {
            'params': all_params,
            'rating_meme': all_prob1,
            'idx': all_idxs,
            'variance_meme': all_variance,
        }
        if args.test_set_is_labeled:
            result['rating_ground_truth'] = all_ratings
            result['variance_ground_truth'] = all_ground_truth_variances
        output_file = os.path.join(args.output_dir, f"result_{accelerator.process_index}.yaml")
        with open(output_file, 'w') as f:
            yaml.dump(result, f)

    if accelerator.is_main_process:
        all_idxs = []
        all_params = []
        all_prob1 = []
        all_variance = []
        all_ratings = []  # Only used if test_set_is_labeled is True
        all_ground_truth_variances = []  # Only used if test_set_is_labeled is True
        for i in range(accelerator.num_processes):
            output_file = os.path.join(args.output_dir, f"result_{i}.yaml")
            with open(output_file, 'r') as f:
                result = yaml.load(f, Loader=yaml.FullLoader)
            all_idxs.extend(result['idx'])
            all_params.extend(result['params'])
            all_prob1.extend(result['rating_meme'])
            all_variance.extend(result['variance_meme'])
            if args.test_set_is_labeled:
                all_ratings.extend(result['rating_ground_truth'])
                all_ground_truth_variances.extend(result['variance_ground_truth'])

        final_result = {
            'params': all_params,
            'rating_meme': all_prob1,
            'idx': all_idxs,
            'variance_meme': all_variance,
        }
        if args.test_set_is_labeled:
            final_result['rating_ground_truth'] = all_ratings
            final_result['variance_ground_truth'] = all_ground_truth_variances
        final_result_indexed_by_idx = {}
        for idx, param, prob1, variance in zip(all_idxs, all_params, all_prob1, all_variance):
            final_result_indexed_by_idx[idx] = {
                'param': param,
                'rating_meme': prob1,
                'variance_meme': variance,
            }
            if args.test_set_is_labeled:
                final_result_indexed_by_idx[idx]['rating_ground_truth'] = all_ratings[all_idxs.index(idx)]
                final_result_indexed_by_idx[idx]['variance_ground_truth'] = all_ground_truth_variances[all_idxs.index(idx)]
        final_output_file = os.path.join(args.output_dir, "final_result_by_idx.yaml")
        with open(final_output_file, 'w') as f:
            yaml.dump(final_result_indexed_by_idx, f)
        final_output_file = os.path.join(args.output_dir, "final_result_as_list.yaml")
        with open(final_output_file, 'w') as f:
            yaml.dump(final_result, f)

        if args.test_set_is_labeled:
            # Compute Kendall Tau and Spearman correlation
            tau, _ = kendalltau(all_ratings, all_prob1)
            spearman_corr, _ = spearmanr(all_ratings, all_prob1)
            print(f"Kendall Tau: {tau}")
            print(f"Spearman Correlation: {spearman_corr}")
            metrics = {
                'rating_kendall_tau': float(tau),
                'rating_spearman_correlation': float(spearman_corr),
            }
            if scoring_model.score_type == 'beta':
                # Variance metrics only make sense for beta
                variance_tau, _ = kendalltau(all_ground_truth_variances, all_variance)
                variance_spearman_corr, _ = spearmanr(all_ground_truth_variances, all_variance)
                print(f"Variance Kendall Tau: {variance_tau}")
                print(f"Variance Spearman Correlation: {variance_spearman_corr}")
                metrics['variance_kendall_tau'] = float(variance_tau)
                metrics['variance_spearman_correlation'] = float(variance_spearman_corr)

            score_file = os.path.join(args.output_dir, "scores.yaml")
            with open(score_file, 'w') as f:
                yaml.dump(metrics, f)


if __name__ == "__main__":
    main()