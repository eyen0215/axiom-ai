import numpy as np

print("=== LE A2 ===")
d_train = np.load('data/linear_elasticity/train_A2.npz')
d_test_b = np.load('data/linear_elasticity/test_scenario_B.npz')
print(f"train feature range: [{d_train['features'].min():.4f}, {d_train['features'].max():.4f}]")
print(f"test_B A2_features range: [{d_test_b['A2_features'].min():.4f}, {d_test_b['A2_features'].max():.4f}]")
print(f"gap between train max and test min: train_max={d_train['features'].max():.4f}")

print("\n=== Maxwell A1 ===")
d_train = np.load('data/maxwell/train_A1.npz')
d_test_a = np.load('data/maxwell/test_scenario_A.npz')
print(f"train feature range: [{d_train['features'].min():.6f}, {d_train['features'].max():.6f}]")
print(f"test_A A1_features range: [{d_test_a['A1_features'].min():.6f}, {d_test_a['A1_features'].max():.6f}]")
print(f"breakdown threshold: 0.05")
print(f"is train always < 0.05? {(d_train['features'] < 0.05).all()}")
print(f"is test always > 0.05? {(d_test_a['A1_features'] > 0.05).all()}")