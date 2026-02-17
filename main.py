from src.train import train_model
from src.logger import append_result_md

def main():
    experiments = [
        ("lr_1e-3", {"lr": 1e-3, "batch_size": 64, "epochs": 10}),
        ("lr_5e-4", {"lr": 5e-4, "batch_size": 64, "epochs": 10}),
        ("lr_1e-4", {"lr": 1e-4, "batch_size": 64, "epochs": 10}),
    ]

    for name, cfg in experiments:
        print(f"\n=== Running experiment: {name} ===")
        model, result = train_model(**cfg)
        append_result_md(result, name)

if __name__ == "__main__":
    main()
