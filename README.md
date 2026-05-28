# EMA-Nesterov Module
A python3 module for [EMA-Nesterov](https://arxiv.org/abs/2605.25395).
## Installation
```
pip install git+https://github.com/OptimAI-Lab/ema-nesterov-module
```
## Usage
You can apply EMA-Nesterov on top of any pytorch optimizer. An example of EMA-Nesterov applied to the AdamW optimizer:
```
import torch
from ema_nesterov import EMA_Nesterov

model = ...
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-3)
optimizer = EMA_Nesterov(
    model.parameters(),
    optimizer,
    lookahead_stepsize=0.5,
    lookahead_ema=0.99,
    warmup_step=1000,
    rest_step=num_iterations - 500,
)
...

for data, target in dataloader:
    optimizer.nesterov_step() # lookahead step, x <- x + beta * m
    loss = model(data, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

- `lookahead_ema=0.99` sets the EMA rate $\gamma=0.99$ in the ema lookahead direction $\mathbf{m}^{t+1} = \gamma\cdot\mathbf{m}^t + (1-\gamma)\cdot (\mathbf{x}^{t+1} - \mathbf{x}^t)$. 
- The above example uses 1000 steps of lookahead warmup by `warmup_step=1000` (i.e., no lookahead at the begining) and 500 steps of lookahead rest by `rest_step=num_iterations - 500` (i.e., no lookahead at the end). 
- `lookahead_stepsize=0.5` sets the maximum lookahead step size $\max_t \beta_t = 0.5$ in the lookahead update $\mathbf{x}^{t+1} = A_t(\mathbf{x}^t + \beta_t \mathbf{m}^t)$ with base optimizer $A_t$, e.g., in the above example,

$$\beta_t = 0 \quad  \quad \quad \quad \quad  \quad ~ \text{for} \quad t\in [0, 1000) \cup [T-500, T],$$
$$\beta_t = 0.5 \cdot \frac{\alpha_t}{ \max_r \alpha_r } \quad \text{for} \quad t\in [1000, T-500),$$

where $\alpha_t$ is the learning rate of optimizer $A_t$ and $T$ is `num_iterations`.

## More Examples
See https://github.com/OptimAI-Lab/ema-nesterov.