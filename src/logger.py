from datetime import datetime

RESULTS_FILE = "results.md"

def append_result_md(result: dict, exp_name: str):
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8"):
            exists = True
    except FileNotFoundError:
        exists = False

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        if not exists:
            f.write("# Experiment results\n\n")
            f.write(
                "| Time | Experiment | Epochs | Batch | LR | "
                "Train acc | Train loss | Test acc | Test loss |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")

        f.write(
            f"| {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {exp_name} "
            f"| {result['epochs']} | {result['batch_size']} | {result['lr']} "
            f"| {result['train_acc']:.3f} | {result['train_loss']:.3f} "
            f"| {result['test_acc']:.3f} | {result['test_loss']:.3f} |\n"
        )
