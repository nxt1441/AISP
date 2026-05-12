"""
Q-Align-AWQ alignment loss.

QAlignLoss adds a regularisation term to standard cross-entropy loss so that,
during backdoor fine-tuning, weight energy is pushed into AWQ-protected channels
(the top ~1% by input-activation magnitude).  After AWQ quantisation those
channels survive with minimal distortion, keeping the backdoor intact.

    L_total = L_CE  +  λ * L_align

    L_align = Σ_layers  mean( (1 - mask) * col_norms_sq )

    col_norms_sq[j] = Σ_i W[i,j]^2    squared L2 norm of weight column j
    mask[j]         = 1  →  AWQ-protected  (no penalty)
                      0  →  AWQ-compressed (penalised)

Usage
-----
    from q_align.loss import QAlignLoss

    q_align_loss = QAlignLoss(masks, lambda_align=0.1)
    total_loss, info = q_align_loss(ce_loss, model)
    total_loss.backward()
"""

import torch
import torch.nn as nn


class QAlignLoss:
    """Combines cross-entropy loss with an AWQ alignment regulariser."""

    def __init__(self, masks: dict, lambda_align: float = 0.1) -> None:
        """Initialise with pre-computed saliency masks.

        Args:
            masks:        Dict mapping layer_name -> binary float tensor [C_in].
                          Produced by SaliencyExtractor.extract_masks().
                          1 = AWQ-protected channel, 0 = unprotected channel.
            lambda_align: Coefficient for the alignment term (default 0.1).
        """
        self.masks        = masks
        self.lambda_align = lambda_align

    def _find_mask(self, layer_name: str) -> torch.Tensor | None:
        """Return the mask whose key overlaps with layer_name, or None.

        Checks containment in both directions to handle PEFT name prefixes
        (e.g. "base_model.model.layers.0.self_attn.q_proj" matching
        "model.layers.0.self_attn.q_proj").

        Args:
            layer_name: Dotted module name from model.named_modules().

        Returns:
            Matching mask tensor, or None if no match is found.
        """
        for key, mask in self.masks.items():
            if key in layer_name or layer_name in key:
                return mask
        return None

    def compute_alignment_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute L_align summed over all matched Linear layers.

        For each nn.Linear layer with a matching mask:
            unprotected  = 1.0 - mask                  [C_in]
            col_norms_sq = (W ** 2).sum(dim=0)          [C_in]  (W is [C_out, C_in])
            layer_loss   = mean(unprotected * col_norms_sq)

        Layers with no matching mask are skipped silently.

        Args:
            model: The model being fine-tuned (plain or PEFT-wrapped).

        Returns:
            Scalar alignment loss tensor on the same device as the model weights.
            Returns zero if no layers were matched.
        """
        alignment_loss = None

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Skip frozen weights — in LoRA, the base layer is frozen so gradients
            # cannot flow through it. Only lora_A matrices are trainable and have
            # shape [r, C_in], whose column norms align with the [C_in] mask.
            if not module.weight.requires_grad:
                continue
            mask = self._find_mask(name)
            if mask is None:
                continue

            mask         = mask.to(module.weight.device)
            unprotected  = 1.0 - mask                       # [C_in]
            col_norms_sq = (module.weight ** 2).sum(dim=0)  # [C_in] for lora_A, [r] for lora_B

            # lora_B weight is [C_out, r] — col_norms gives [r], not [C_in]; skip it.
            if col_norms_sq.shape != unprotected.shape:
                continue

            layer_loss   = (unprotected * col_norms_sq).mean()

            alignment_loss = layer_loss if alignment_loss is None else alignment_loss + layer_loss

        if alignment_loss is None:
            device = next(model.parameters()).device
            alignment_loss = torch.tensor(0.0, device=device)

        return alignment_loss

    def __call__(
        self,
        ce_loss: torch.Tensor,
        model: nn.Module,
    ) -> tuple:
        """Return total loss and a breakdown dict.

        Args:
            ce_loss: Cross-entropy loss from the model forward pass.
            model:   The model being fine-tuned.

        Returns:
            total_loss: Scalar tensor = ce_loss + lambda_align * L_align.
            info:       Dict with keys "ce", "align", "total" (Python floats).
        """
        alignment_loss = self.compute_alignment_loss(model)
        total_loss     = ce_loss + self.lambda_align * alignment_loss

        info = {
            "ce":    ce_loss.item(),
            "align": alignment_loss.item(),
            "total": total_loss.item(),
        }
        return total_loss, info
