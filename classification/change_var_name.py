import torch

dst = "/home/hke3/official_icml/Deformba/classification/tiny_ckpt.pth"

ckpt = torch.load(dst, map_location="cpu")

def rename_state_dict(state):
    new_state = {}
    for k, v in state.items():
        new_k = k

        # 示例：如果你把 da_scan 改名成 dynamic_scan
        new_k = new_k.replace("token_mixer.da_scan.", "token_mixer.CASF.")

        # 示例：如果你把 offset 改名成 offset_layer
        # new_k = new_k.replace(".offset.", ".offset_layer.")

        new_state[new_k] = v
    return new_state

# 常见 checkpoint 结构
if "model" in ckpt:
    state_dict = ckpt["model"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print("Number of model keys:", len(state_dict))

for k in state_dict.keys():
    print(k)

if "model" in ckpt:
    ckpt["model"] = rename_state_dict(ckpt["model"])

if "model_ema" in ckpt:
    ckpt["model_ema"] = rename_state_dict(ckpt["model_ema"])

torch.save(ckpt, dst)
print("saved to", dst)
