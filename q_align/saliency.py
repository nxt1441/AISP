"""
AWQ saliency mask extraction for Q-Align-AWQ.

SaliencyExtractor runs the clean model on a calibration corpus, records
per-channel input-activation magnitudes with forward hooks, then returns
a binary mask marking which channels AWQ will protect (top 1% by default).

Usage
-----
    from q_align.saliency import SaliencyExtractor

    extractor = SaliencyExtractor()
    masks = extractor.extract_masks(model_path, calib_texts, top_percent=0.01)
    extractor.save_masks(masks, "saliency_masks.pt")
    extractor.print_stats(masks)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class SaliencyExtractor:
    """Computes AWQ protection masks from calibration-corpus activation statistics."""

    def extract_masks(
        self,
        model_path: str,
        calib_texts: list,
        top_percent: float = 0.01,
    ) -> dict:
        """Run the clean model on calib_texts and return binary AWQ protection masks.

        Saliency score for input channel j of a Linear layer:
            S_j = (1/N) * Σ_i |x_j^(i)|      mean absolute activation over tokens

        Protection threshold:
            τ       = percentile(S, 99)        99th-percentile by default
            mask[j] = 1  if S_j >= τ           AWQ will protect this channel
                      0  otherwise              AWQ will compress/distort this channel

        Hooks are always removed in a finally block to prevent memory leaks.

        Args:
            model_path:   Path to the base model directory (bfloat16, device_map=auto).
            calib_texts:  Plain-text calibration strings; should not overlap training data.
            top_percent:  Fraction of channels to mark as protected (default 0.01 = 1%).

        Returns:
            Dict mapping layer_name -> float tensor of shape [C_in] with values in {0, 1}.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        # accumulated[layer_name] = list of per-sample mean-abs tensors, each [C_in]
        accumulated: dict = {}
        hooks = []

        def _make_hook(name: str):
            def _hook(module, inp, _out):
                x = inp[0]                              # [batch, seq, C_in]
                x_2d = x.reshape(-1, x.shape[-1])      # [batch*seq, C_in]
                mean_abs = x_2d.abs().mean(dim=0).detach().cpu().float()
                accumulated.setdefault(name, []).append(mean_abs)
            return _hook

        try:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    hooks.append(module.register_forward_hook(_make_hook(name)))

            with torch.no_grad():
                for text in calib_texts:
                    enc = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(model.device)
                    model(**enc)
        finally:
            for h in hooks:
                h.remove()

        masks = {}
        percentile_q = 1.0 - top_percent          # 0.99 for top 1%
        for name, tensors in accumulated.items():
            S   = torch.stack(tensors).mean(dim=0) # [C_in], mean over calibration samples
            tau = torch.quantile(S, percentile_q)
            masks[name] = (S >= tau).float()

        return masks

    def save_masks(self, masks: dict, path: str) -> None:
        """Persist saliency masks to disk with torch.save.

        Args:
            masks: Dict of layer_name -> mask tensor.
            path:  Destination file path (e.g. "output/saliency_masks.pt").
        """
        torch.save(masks, path)
        print(f"Masks saved → {path}  ({len(masks)} layers)")

    def load_masks(self, path: str) -> dict:
        """Load saliency masks previously saved with save_masks.

        Args:
            path: Path to the .pt file produced by save_masks.

        Returns:
            Dict of layer_name -> mask tensor.
        """
        return torch.load(path, weights_only=True)

    def print_stats(self, masks: dict) -> None:
        """Print per-layer and overall protection statistics.

        Args:
            masks: Dict of layer_name -> binary mask tensor [C_in].
        """
        total_ch = 0
        prot_ch  = 0
        col_w    = 60

        print(f"\n{'Layer':<{col_w}} {'C_in':>6}  {'Protected':>9}  {'%':>5}")
        print("-" * (col_w + 28))
        for name, mask in masks.items():
            c_in = mask.numel()
            prot = int(mask.sum().item())
            total_ch += c_in
            prot_ch  += prot
            pct = prot / c_in * 100 if c_in else 0
            label = name if len(name) <= col_w else "…" + name[-(col_w - 1):]
            print(f"  {label:<{col_w - 2}} {c_in:>6,}  {prot:>9,}  {pct:>4.1f}%")
        print("-" * (col_w + 28))
        overall = prot_ch / total_ch * 100 if total_ch else 0
        print(f"  {'TOTAL':<{col_w - 2}} {total_ch:>6,}  {prot_ch:>9,}  {overall:>4.1f}%\n")
