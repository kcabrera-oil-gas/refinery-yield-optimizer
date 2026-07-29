import numpy as np
import pandas as pd

def generate_dataset(n_samples=1000, seed=42):
    np.random.seed(seed)
    n = n_samples
    f1 = np.random.uniform(0, 100, n)
    f2 = np.random.uniform(0, 100, n)  
    f3 = np.random.uniform(0, 100, n)
    f4 = np.random.uniform(0, 100, n)
    f5 = np.random.uniform(0, 100, n)
    f6 = np.random.uniform(0, 100, n)
    f7 = np.random.uniform(0, 100, n)
    f8 = np.random.uniform(0, 100, n)
    target = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8) / 8 + np.random.normal(0, 2, n)
    df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5, "f6": f6, "f7": f7, "f8": f8, "target": target})
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
    print(f"Generated {len(df)} samples")
