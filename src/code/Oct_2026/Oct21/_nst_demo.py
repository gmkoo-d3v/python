import json, time, os
os.chdir(r"C:\code\src\code\Oct_2026\Oct20")  # 노트북이 '1.png' 등을 상대경로로 열기 때문
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB = r"C:\code\src\code\Oct_2026\Oct20\main.ipynb"
nb = json.load(open(NB, encoding="utf-8"))

def cell(cid):
    for c in nb["cells"]:
        if c["id"] == cid:
            return "".join(c["source"])
    raise KeyError(cid)

ns = {}
for cid in ["img-load-simple", "fix2", "fix3", "fix5"]:
    exec(cell(cid), ns)

content = ns["content"]; style = ns["style"]
vgg = ns["vgg"]
get_features = ns["get_features"]; gram_matrix = ns["gram_matrix"]
content_features = ns["content_features"]; style_grams = ns["style_grams"]
im_convert = ns["im_convert"]
device = ns["device"]

style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
content_weight, style_weight = 1, 1e7

target = content.clone().requires_grad_(True).to(device)
optimizer = torch.optim.Adam([target], lr=0.01)
steps = 1200
t0 = time.time()
for ii in range(1, steps + 1):
    tf = get_features(target, vgg)
    content_loss = torch.mean((tf['conv4_2'] - content_features['conv4_2']) ** 2)
    style_loss = 0
    for layer in style_weights:
        f = tf[layer]; g = gram_matrix(f); _, d, h, w = f.shape
        style_loss += style_weights[layer] * torch.mean((g - style_grams[layer]) ** 2) / (d * h * w)
    total = content_weight * content_loss + style_weight * style_loss
    optimizer.zero_grad(); total.backward(); optimizer.step()
    if ii % 100 == 0 or ii == 1:
        print(f"step {ii:4d} | content {content_loss.item():.4f} | style {style_loss.item():.4f} | el {(time.time()-t0)/ii:.3f}s/it", flush=True)

out = r"C:\code\src\code\Oct_2026\Oct21\style_result.png"
plt.imsave(out, im_convert(target))

fig, axes = plt.subplots(3, 1, figsize=(8, 12))
axes[0].imshow(im_convert(content)); axes[0].set_title("Content"); axes[0].axis("off")
axes[1].imshow(im_convert(style)); axes[1].set_title("Style"); axes[1].axis("off")
axes[2].imshow(im_convert(target)); axes[2].set_title("Target (styled)"); axes[2].axis("off")
plt.tight_layout()
fig.savefig(r"C:\code\src\code\Oct_2026\Oct21\style_compare.png", dpi=120)
print("saved", out, flush=True)
