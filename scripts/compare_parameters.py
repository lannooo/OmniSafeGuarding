import torch
from transformers import Qwen2_5OmniForConditionalGeneration


model_before = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "models/base/Qwen2.5-Omni-7B"
)
model_after = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "models/guard/OmGuard-Qwen2.5-7B-Enhance"
)

updated_params = {}
total_params_before = 0
total_params_after = 0

# Compute total parameter counts before and after fine-tuning (optional)
for name, param in model_before.named_parameters():
    total_params_before += param.numel()
for name, param in model_after.named_parameters():
    total_params_after += param.numel()

if total_params_before != total_params_after:
    print(f"Warning: Total parameters changed from {total_params_before} to {total_params_after}. This might indicate structural changes.")

# 2. Compare parameters one by one
for (name_before, param_before), (name_after, param_after) in zip(
    model_before.named_parameters(), model_after.named_parameters()
):
    if name_before != name_after:
        raise ValueError(f"Parameter names do not match: {name_before} vs {name_after}")

    # 3. Compare parameter values
    if not torch.equal(param_before.data, param_after.data): 
        # Parameter value has changed
        updated_params[name_before] = param_after.numel() # Record parameter name and size

# 4. Aggregate updated parameters by module
updated_modules = {}
for param_name, param_size in updated_params.items():
    # Parse module name (e.g., from 'model.layers.0.self_attn.q_proj.weight' to 'model.layers.0.self_attn')
    # Use '.' as the separator and take all parts except the last
    parts = param_name.split('.')
    module_name = '.'.join(parts[:-1]) # Drop the last part (usually the parameter name, e.g., weight or bias)

    if module_name not in updated_modules:
        updated_modules[module_name] = 0
    updated_modules[module_name] += param_size

# 5. Print results
print("Changed modules and their updated parameter counts after fine-tuning:")
for module_name, num_params_updated in updated_modules.items():
    print(f"  - Module: {module_name}, Updated parameters: {num_params_updated}")
print(f"\nTotal updated parameters: {sum(updated_modules.values())}")
print(f"Total model parameters: {total_params_before} (before fine-tuning) / {total_params_after} (after fine-tuning)")
