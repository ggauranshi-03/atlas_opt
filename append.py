markdown_table = """
### Synthetic Experiment Results (Best Hyperparameters)

| Algorithm | Alpha | Best LR | Best Rho | Mean Final Loss $F(W_t)$ | Std Final Loss | Mean Dist to $W^*$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Atlas | 1.2 | 0.005 | 0.01 | 0.1058 | 0.0053 | 11.590 |
| Atlas Random | 1.2 | 0.005 | 0.01 | 0.1058 | 0.0053 | 11.590 |
| Atlas Raw | 1.2 | 0.005 | 0.01 | 0.1058 | 0.0053 | 11.590 |
| Muon-SAM | 1.2 | 0.005 | 0.05 | 0.0977 | 0.0005 | 10.756 |
| Atlas | 1.6 | 0.005 | 0.01 | 0.1018 | 0.0024 | 11.188 |
| Atlas Random | 1.6 | 0.005 | 0.02 | 0.1018 | 0.0024 | 11.188 |
| Atlas Raw | 1.6 | 0.005 | 0.01 | 0.1018 | 0.0024 | 11.188 |
| FSAM | 1.6 | 0.003 | 0.02 | 1.2906 | 0.0181 | 96.301 |
| Muon-SAM | 1.6 | 0.005 | 0.05 | 0.0933 | 0.0004 | 10.317 |
| Atlas | 2.0 | 0.005 | 0.01 | 0.0980 | 0.0010 | 10.804 |
| Atlas Random | 2.0 | 0.005 | 0.01 | 0.0980 | 0.0010 | 10.804 |
| Atlas Raw | 2.0 | 0.005 | 0.02 | 0.0980 | 0.0010 | 10.804 |
| FSAM | 2.0 | 0.003 | 0.05 | 0.3728 | 0.0147 | 36.080 |
| Muon-SAM | 2.0 | 0.005 | 0.05 | 0.0896 | 0.0004 | 9.941 |
"""

with open('README.md', 'a', encoding='utf-8') as f:
    f.write(markdown_table)
