import torch
from torch.nn.functional import relu
from . import _mmhdc_cpp

class MultiMMHDC(torch.nn.Module):
    def __init__(self, num_classes: int, 
                 out_channels: int, 
                 lr: float = 1e-2, 
                 C: float = 1.0, 
                 device: str = 'cpu',
                 backend: str = 'python',
                 dtype: torch.dtype = torch.float32):
        
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.lr = lr
        self.device = device
        self.C = C
        self.backend = backend
        self.prototypes = torch.nn.parameter.Parameter(
            data=torch.zeros(num_classes, out_channels, dtype=dtype, device=device), 
            requires_grad=False
        )
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        return torch.argmax(x @ self.prototypes.T, dim=1)
    
    # Initialization of prototypes
    def initialize(self, x: torch.Tensor, y: torch.Tensor):
        for i in range(self.num_classes):
            self.prototypes[i] = torch.mean(x[y.squeeze() == i], 0).to(self.device)

        # Normalizing prototypes
        eps = 1e-8 * torch.ones(self.prototypes.size(0), 1, device=self.prototypes.device, dtype=self.dtype)
        self.prototypes /= torch.maximum(eps, torch.norm(self.prototypes, dim=1, keepdim=True))

    def loss(self, X: torch.Tensor, y: torch.Tensor):
        loss = torch.pow(torch.norm(self.prototypes, dim=-1), 2).sum() / (2 * self.C)
        for cls1 in torch.unique(y):
            loss += torch.sum(relu(2 - X[y == cls1] @ (self.prototypes[cls1] - self.prototypes).T))

        return loss
    
    def step(self, x: torch.Tensor, y: torch.Tensor):
        if self.backend == 'cpp':
            self.prototypes.data = _mmhdc_cpp.step(x, y, self.prototypes, self.lr, self.C)
        elif self.backend == 'python':
            self._py_step(x, y)
        else:
            raise ValueError(f"Unsupported backend '{self.backend}'. Expected 'cpp' or 'python'.")

    def _py_step(self, x: torch.Tensor, y: torch.Tensor, optimized=True):
        # This implementation is for reference purpose only. Use matrix-based optimized C++ and Python implementations
        def _py_step_reference(x: torch.Tensor, y: torch.Tensor):
            prototypes_update = torch.zeros_like(self.prototypes, dtype=self.dtype)
            for cls in y.unique():
                rolled_prototypes = torch.roll(self.prototypes, -cls.item(), dims=0)
                x_cls = x[y == cls]

                dot = x_cls @ (rolled_prototypes[0] - rolled_prototypes[1:]).T
                hinge_loss = relu(2 - dot)

                exceeding_margin = hinge_loss > 0
                num_violations = exceeding_margin.sum(dim=1, dtype=x_cls.dtype)
                prototypes_update[cls] += (x_cls * num_violations.unsqueeze(1)).sum(0)

                y_true_all = exceeding_margin.any(0).nonzero().flatten()
                for y_true in y_true_all:
                    prototypes_update[(y_true + 1 + cls) % self.num_classes] -= x_cls[exceeding_margin[:, y_true]].sum(0)
            
            self.prototypes.data = (1 - self.lr / self.C) * self.prototypes.data + self.lr * prototypes_update
        
        # Optimized code based on the C++ implementation logic
        def _py_step_optimized(x: torch.Tensor, y: torch.Tensor):
            scores = x @ self.prototypes.T
            correct_scores = scores.gather(1, y.unsqueeze(1))

            violated = (correct_scores - scores) < 2
            violated.scatter_(1, y.unsqueeze(1), False)

            W = -violated.to(x.dtype)
            W.scatter_add_(1, y.unsqueeze(1), violated.sum(dim=1, keepdim=True).to(x.dtype))

            prototypes_update = W.T @ x
            self.prototypes.data = (1 - self.lr / self.C) * self.prototypes.data + self.lr * prototypes_update

        _py_step_reference(x, y) if not optimized else _py_step_optimized(x, y)
