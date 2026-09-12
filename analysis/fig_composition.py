"""Build the composition figure from manuscript_numbers.json. Nothing is re-aggregated here;
the json is read as-is, so the figure cannot disagree with the tables."""

import os as _os
# Results live at results/paper/<run>/rollouts.jsonl, already merged across the machines
# they were collected on (see docs/RESULTS_INDEX.md for the merge rule).
ROOT = _os.environ.get("LIBERO_CTRL_ROOT",
       _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
RESULTS = _os.path.join(ROOT, "results", "paper")
MANIFESTS = _os.path.join(ROOT, "manifests", "v0.1")
OUT_DIR = _os.path.join(ROOT, "analysis", "out")
_os.makedirs(OUT_DIR, exist_ok=True)

import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D=json.load(open("analysis/manuscript_numbers.json"))
ORDER=["MINERVA","SmolVLA","VLA-JEPA","pi05","OpenVLA-OFT","UniVLA"]
LAB={"MINERVA":"MINERVA","SmolVLA":"SmolVLA","VLA-JEPA":"VLA-JEPA","pi05":r"$\pi_{0.5}$",
     "OpenVLA-OFT":"OpenVLA-OFT","UniVLA":"UniVLA"}
COL={"MINERVA":"#4C72B0","SmolVLA":"#55A868","VLA-JEPA":"#C44E52","pi05":"#8172B2",
     "OpenVLA-OFT":"#CCB974","UniVLA":"#64B5CD"}
MK={"L1":"o","L2":"s","L3":"^"}
FLOOR=10.0   # below this predicted value the residual is not resolvable (same criterion as the paper)
fig,axs=plt.subplots(1,2,figsize=(11.5,4.9))
ax=axs[0]
ax.plot([0,100],[0,100],"k--",lw=1,alpha=.55,zorder=0)
ax.text(52,44,"$y=x$",rotation=45,ha="center",va="center",fontsize=9,alpha=.7)
for m in ORDER:
    r=D[m]; partial = r["cells"]<40
    for L in ("L1","L2","L3"):
        d=r["prod"][L]; p,o=d["pred"],d["obs"]
        floored = p<FLOOR
        ax.errorbar(p,o,yerr=[[max(0,o-(p+d["lo"]))],[max(0,(p+d["hi"])-o)]],
                    fmt=MK[L],ms=8.5,color=COL[m],
                    mfc="white" if (partial or floored) else COL[m],mew=1.7,
                    ecolor=COL[m],elinewidth=1.1,capsize=2.5,alpha=.92,zorder=3)
ax.set_xlabel("Predicted by product of single-axis factors (\\%)" if False else
              "Predicted six-factor success rate (%)")
ax.set_ylabel("Observed six-factor success rate (%)")
ax.set_title("(a) Rate-level composition",fontsize=11)
ax.set_xlim(-4,101); ax.set_ylim(-4,101); ax.grid(alpha=.22)
ax=axs[1]
ax.axhline(0,color="k",ls="--",lw=1,alpha=.55,zorder=0)
xs=[];ys=[]
for m in ORDER:
    r=D[m]; partial = r["cells"]<40
    for L in ("L1","L2","L3"):
        d=r["prod"][L]; p,dev=d["pred"],d["dev"]
        floored = p<FLOOR
        if not floored: xs.append(p); ys.append(dev)
        ax.errorbar(p,dev,yerr=[[max(0,dev-d["lo"])],[max(0,d["hi"]-dev)]],
                    fmt=MK[L],ms=8.5,color=COL[m],
                    mfc="white" if (partial or floored) else COL[m],mew=1.7,
                    ecolor=COL[m],elinewidth=1.1,capsize=2.5,alpha=.92,zorder=3)
sl,ic=np.polyfit(xs,ys,1)
xx=np.linspace(min(xs),100,50)
ax.plot(xx,sl*xx+ic,color="0.35",lw=1.3,ls=":",zorder=1)
ax.text(0.97,0.05,f"slope {sl:+.3f} pp/pp  ($n={len(xs)}$)",transform=ax.transAxes,
        ha="right",va="bottom",fontsize=9,color="0.3")
ax.axvspan(-4,FLOOR,color="0.85",alpha=.45,zorder=0)
ax.text(3,ax.get_ylim()[1]*0.92,"floored",fontsize=8,color="0.45",rotation=90,va="top")
ax.set_xlabel("Predicted six-factor success rate (%)")
ax.set_ylabel("Deviation: observed $-$ predicted (pp)")
ax.set_title("(b) Residual",fontsize=11)
ax.set_xlim(-4,101); ax.grid(alpha=.22)
h=[plt.Line2D([],[],color=COL[m],marker="o",ls="",ms=8,label=LAB[m]) for m in ORDER]
h+=[plt.Line2D([],[],color="0.35",marker=MK[L],ls="",ms=8,label=L) for L in ("L1","L2","L3")]
h+=[plt.Line2D([],[],color="0.35",marker="o",ls="",ms=8,mfc="white",mew=1.7,
               label="incomplete / floored")]
fig.legend(handles=h,loc="lower center",ncol=5,frameon=False,fontsize=9,bbox_to_anchor=(0.5,-0.10))
fig.tight_layout(rect=[0,0.07,1,1])
for p in (f"{OUT_DIR}/composition.png", f"{OUT_DIR}/composition.pdf"):
    fig.savefig(p,dpi=200,bbox_inches="tight")
print(f"saved -> {OUT_DIR}/composition.png")
print(f"residual slope (floor cells excluded, n={len(xs)}): {sl:+.4f} pp/pp")
