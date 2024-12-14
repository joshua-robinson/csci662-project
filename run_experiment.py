import argparse
import os


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment_type",
        help="Type of experiment",
        choices=["kv", "local", "scattered"],
        required=True
    )
    parser.add_argument(
        "-b",
        "--bound_ratio",
        help="Bound ratio for compression",
        required=True
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of model to use",
        default="togethercomputer/RedPajama-INCITE-Base-3B-v1"
    )
    parser.add_argument(
        "-t",
        "--train_file",
        help="Name of file to use for training data",
        default="wikitext2/train.json"
    )
    parser.add_argument(
        "-v",
        "--validation_file",
        help="Name of file to use for validation data",
        default="wikitext2/test.json"
    )
    parser.add_argument(
        "-l",
        "--max_span_len",
        help="Max span length for KV compression",
        default=25
    )
    parser.add_argument(
        "-p",
        "--pos_ids_by_doc",
        help="Position ids by document, not block",
        action="store_true"
    )
    parser.add_argument(
        "-x",
        "--mask_across_docs",
        help="Mask across documents",
        action="store_true"
    )
    parser.add_argument(
        "-g",
        "--no_grad_accum",
        help="Don't use gradient accumulation",
        action="store_true"
    )
    parser.add_argument(
        "-s",
        "--strategic_span_selection",
        help="Selects spans to wrap in <CL> and <CR> strategically",
        action="store_true"
    )
    args = parser.parse_args()
    return args


def make_command_from_args(args: argparse.Namespace) -> str:
    if args.experiment_type == "local":
        type_str = "local_attention"
    elif args.experiment_type == "scattered":
        type_str = "scattered_attention"
    elif args.experiment_type == "kv":
        type_str = "kv_compression"
    else:
        raise ValueError("Invalid type")
    cmnd = f"CUDA_VISIBLE_DEVICES=0, python run_clm_{type_str}.py "
    cmnd += f"--model_name_or_path {args.model} "
    cmnd += f"--train_file {args.train_file} "
    cmnd += f"--validation_file {args.validation_file} "
    if args.no_grad_accum:
        cmnd += "--per_device_train_batch_size 12 "
        cmnd += "--gradient_accumulation_steps 1 "
        cmnd += "--per_device_eval_batch_size 12 "
    else:
        cmnd += "--per_device_train_batch_size 4 "
        cmnd += "--gradient_accumulation_steps 3 "
        cmnd += "--per_device_eval_batch_size 4 "
    cmnd += "--block_size 256 "
    cmnd += "--preprocessing_num_workers 12 "
    output_dir = f"./output/{args.train_file}/{args.validation_file}/{args.model}/"
    output_dir += f"{args.experiment_type}_{args.bound_ratio}"
    if args.pos_ids_by_doc:
        output_dir += "_pos_ids_by_doc"
        cmnd += "--pos_ids_by_doc "
    if args.mask_across_docs:
        output_dir += "_mask_across_docs"
        cmnd += "--mask_across_docs "
    if args.no_grad_accum:
        output_dir += "_no_grad_accum"
    if args.strategic_span_selection:
        if args.experiment_type != "kv":
            raise ValueError("Strategic span selection only works with experiment type 'kv'")
        output_dir += "_strategic_span_selection"
        cmnd += "--strategic_selection "
    if args.experiment_type == "kv":
        output_dir += f"_{args.max_span_len}"
        cmnd += f"--max_span_length {args.max_span_len} "
    cmnd += f"--output_dir {output_dir} "
    cmnd += "--compress "
    cmnd += f"--bound_ratio {args.bound_ratio} "
    cmnd += "--r 16 "
    cmnd += "--seed 0 "
    cmnd += "--learning_rate 2e-5"
    return cmnd


if __name__ == "__main__":
    args = get_args()
    cmnd = make_command_from_args(args=args)
    os.system(cmnd)
