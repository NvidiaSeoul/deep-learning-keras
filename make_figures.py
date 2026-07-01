"""results/ EDA 시각화 재현 — pandas + matplotlib (TensorFlow 불필요)
신경망 분류 모델이 학습하는 실제 데이터(Titanic, Fish)를 시각화.
실행: python make_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"      # 한글 라벨 (Windows)
plt.rcParams["axes.unicode_minus"] = False
GREEN = "#76B900"
D = os.path.join(os.path.dirname(__file__), "src", "20260611")
R = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(R, exist_ok=True)

# Titanic 이진분류 타깃 탐색
t = pd.read_csv(os.path.join(D, "titanic_passengers.csv"))
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
t.groupby("gender")["Survived"].mean().plot(kind="bar", color=["#4C72B0", GREEN], ax=axs[0])
axs[0].set(title="성별 생존율 (Titanic)", ylabel="생존율", ylim=(0, 1)); axs[0].tick_params(axis="x", rotation=0)
t.groupby("Pclass")["Survived"].mean().plot(kind="bar", color=GREEN, ax=axs[1])
axs[1].set(title="객실 등급별 생존율", ylabel="생존율", ylim=(0, 1)); axs[1].tick_params(axis="x", rotation=0)
fig.suptitle("Titanic 이진분류 — 타깃(Survived) 탐색")
fig.tight_layout(); fig.savefig(f"{R}/titanic_eda.png", dpi=120); plt.close(fig)

# Fish 다중분류 특성 분포
f = pd.read_csv(os.path.join(D, "fish_data.csv"))
fig, ax = plt.subplots(figsize=(6.5, 5))
for sp, g in f.groupby("Species"):
    ax.scatter(g["Length"], g["Weight"], label=sp, s=28, alpha=.8)
ax.set(xlabel="Length", ylabel="Weight", title="어종별 길이–무게 분포 (7-class, Fish)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{R}/fish_species.png", dpi=120); plt.close(fig)
print("saved figures to", R)
