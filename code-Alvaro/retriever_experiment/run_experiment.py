from evaluate import run_evaluation
import json
from generate_dataset import generate_evaluation_dataset

if __name__ == "__main__":
    for k in [1, 3, 5, 10, 15]:
        print(f"\n{'='*50}")
        print(f"Generating dataset with k={k}")
        print(f"{'='*50}")
        
        data = generate_evaluation_dataset(n=100, k=k)
        
        if data:
            dataset_path = f"datasets/dataset_k{k}.json"
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Dataset k={k} saved.")
            
            run_evaluation(
                dataset_path=dataset_path,
                output_csv=f"results/evaluation_k{k}.csv",
                sample_percentage=0.20
            )