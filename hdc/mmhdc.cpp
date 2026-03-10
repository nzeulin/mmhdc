#include <torch/extension.h>

/*
Optimised step procedure:
  1. Pre-sort x/y by class label once → O(1) contiguous slice per class,
     eliminates K boolean masks and K nonzero() calls.
  2. Build wrong-prototype diff matrix directly (no roll, no extra allocation).
  3. Fuse correct-class update: any_violated @ x_cls (one GEMV).
  4. Vectorise inner wrong-class loop: mask_f.t() @ x_cls (one GEMM)
     + a single index_add_ scatter — replaces the serial per-wrong-class loop.
*/

torch::Tensor step(torch::Tensor &x, torch::Tensor &y, torch::Tensor &prototypes, float lr, float C) {
    auto num_classes = prototypes.size(0);
    auto prototypes_update = torch::zeros_like(prototypes);
    auto device = x.device();

    // Pre-sort by class label so each class occupies a contiguous slice.
    auto sorted_idx = torch::argsort(y);
    auto x_sorted   = x.index_select(0, sorted_idx);
    auto y_sorted   = y.index_select(0, sorted_idx);

    // Compute per-class start/end offsets via bincount.
    auto counts  = torch::bincount(y_sorted, /*weights=*/{}, /*minlength=*/num_classes);
    auto offsets = torch::zeros(num_classes + 1,
                                torch::TensorOptions().dtype(torch::kInt64).device(device));
    offsets.slice(0, 1) = torch::cumsum(counts, 0);

    // Index vector [0, 1, ..., K-1] reused every iteration.
    auto all_cls_idx = torch::arange(num_classes,
                                     torch::TensorOptions().dtype(torch::kInt64).device(device));

    for (int64_t cls = 0; cls < num_classes; ++cls) {
        auto start = offsets[cls].item<int64_t>();
        auto end   = offsets[cls + 1].item<int64_t>();
        if (start == end) continue;  // class absent from this batch

        // Contiguous slice — no scatter-gather.
        auto x_cls = x_sorted.slice(0, start, end);  // (N_cls, D)

        // Wrong-prototype indices in the original prototype matrix.
        auto wrong_indices = all_cls_idx.masked_select(all_cls_idx != cls);  // (K-1,)

        // Build pairwise diff: p_correct - p_wrong_k for each k. No roll needed.
        auto p_wrong = prototypes.index_select(0, wrong_indices);            // (K-1, D)
        auto diff    = prototypes[cls].unsqueeze(0) - p_wrong;               // (K-1, D)

        // dot_product[i, k] = x_cls[i] · (p_correct - p_wrong_k)
        auto dot_product      = torch::mm(x_cls, diff.t());                  // (N_cls, K-1)
        auto exceeding_margin = torch::relu(2 - dot_product) > 0;            // (N_cls, K-1) bool
        auto mask_f           = exceeding_margin.to(x.scalar_type());        // (N_cls, K-1) float

        // Correct-class update: sum x_cls rows where any margin is violated.
        // any_violated[i] = 1 iff sample i violates at least one margin.
        auto any_violated = exceeding_margin.any(1).to(x.scalar_type());     // (N_cls,) float
        prototypes_update[cls] += torch::mv(x_cls.t(), any_violated);        // (D,)

        // Wrong-class updates: one GEMM replaces the serial inner loop.
        // neg_updates[k] = sum of x_cls rows where exceeding_margin[:, k] is true.
        auto neg_updates = torch::mm(mask_f.t(), x_cls);                     // (K-1, D)
        prototypes_update.index_add_(0, wrong_indices, -neg_updates);
    }

    // Regularised prototype update.
    prototypes = (1 - lr / C) * prototypes + lr * prototypes_update;
    return prototypes;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("step", &step, "MM-HDC prototype update function");
}
