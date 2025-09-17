import torch
import torch.nn as nn

torch.manual_seed(13)

N = nn.Sequential(
    nn.Linear(10, 10, bias=False),
    nn.ReLU(),
    nn.Linear(10, 10, bias=False),
    nn.ReLU(),
    nn.Linear(10, 3, bias=False)
)

# original input
x = torch.rand((1,10))
x.requires_grad_()

print("Original logits:", N(x).detach().numpy())
print("Original class:", int(N(x).argmax(dim=1).item()))

# target
t = 1

# loss
L = nn.CrossEntropyLoss()

# the eps list to try
eps_list = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 6e-1, 7e-1, 8e-1, 81e-2, 82e-2, 83e-2, 84e-2,85e-2, 9e-1,1.0]


for eps in eps_list:
    N.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    loss = L(N(x), torch.tensor([t], dtype=torch.long))
    loss.backward()

    with torch.no_grad():
        adv_x = (x - eps * x.grad.sign()).clamp(0.0, 1.0)

    with torch.no_grad():
        new_logits = N(adv_x)
        new_class = int(new_logits.argmax(dim=1).item())
        linf = float(torch.norm((x - adv_x).flatten(), p=float('inf')).item())


    print(f"eps={eps: .6f} -> new_class={new_class}, linf={linf:.6f}")



