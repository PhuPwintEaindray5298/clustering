from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet
from scipy.spatial.distance import pdist
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

DATA = Path(__file__).parent.parent / "CLUSTERING" / "data"

OUT = Path(__file__).parent.parent / "CLUSTERING" / "outputs"

OUT.mkdir(exist_ok=True, parents=True)

RNG = np.random.default_rng(42)

plt.rcParams.update({"figure.dpi":130,"font.size":9})

#feature:
def circadian_features(hourly:pd.Series) -> dict:
    x = hourly.values
    n = len(x)
    profile = hourly.groupby(hourly.index.hour).mean()
    p = len(profile)
    xbar = x.mean()
    IS = (n*((profile-xbar)**2).sum()) / (p*((x-xbar)**2).sum())
    IV = (n*(np.diff(x)**2).sum()) / ((n-1)*((x-xbar)**2).sum())
    prof_wrapped = np.concatenate([profile.values,profile.values])
    roll10 = [prof_wrapped[i:i + 10].mean() for i in range(24)]
    roll5 = [prof_wrapped[i:i + 5].mean() for i in range(24)]
    M10,L5 = max(roll10), min(roll5)
    RA = (M10-L5) / (M10+L5)
    return dict(IS=IS, IV=IV, RA=RA, M10=np.log1p(M10), L5=np.log1p(L5))

def person_features(path:Path, n_days:int) -> dict:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    counts = df.groupby(df.index.date).size()
    full_days = counts[counts==1440].index[:n_days]
    df = df[np.isin(df.index.date, full_days)]
    a = df["activity"]
    daily = a.groupby(a.index.date).sum()
    hourly = a.resample("1h").mean().dropna()
    night = a[(a.index.hour >= 0) & (a.index.hour < 6)]
    day = a[(a.index.hour >=8) & (a.index.hour < 22)]
    active = (day > 0).astype(int).values
    #fragmentaton: P(active minute is followed by a rest minute during daytime)
    trans = ((active[:-1] == 1) & (active[1:]==0)).sum() / max(active[:-1].sum(),1)
    f = dict(
        n_days=len(full_days),
        mean_activity=np.log1p(a).mean(),
        daily_cv=daily.std() / daily.mean(),
        pct_inactive=(a == 0).mean(),
        night_share=night.sum() / a.sum(),
        day_fragmentation=trans,
    )
    f.update(circadian_features(hourly=hourly))
    return f

scores = pd.read_csv(DATA/"scores.csv")
rows = []
for _, r in scores.iterrows():
    grp = r["number"].split("_")[0]
    f = person_features(DATA / grp / f"{r['number']}.csv", int(r["days"]))
    f.update(id = r["number"], group = "Depressed" if grp == "condition" else "Control",
            sex="F" if r["gender"] == 1 else "M",
            age=int(str(r["age"]).split("-")[0]) + 2,
            madrs = pd.to_numeric(r["madrs1"],errors="coerce"))
    rows.append(f)
D = pd.DataFrame(rows).set_index("id")
FEATS = ["mean_activity","daily_cv","pct_inactive","night_share","day_fragmentation","IS","IV","RA","M10","L5"]
D.to_csv(OUT/"features.csv")
print(f"Participants: {len(D)} ({(D.group == 'Depressed').sum()} depressed,"
      f"{(D.group=='Control').sum()} controls); days/person median {D.n_days.median():.0f}")

#PCA:
X = StandardScaler().fit_transform(D[FEATS])
pca_full = PCA().fit(X)
evr = pca_full.explained_variance_ratio_
eig = pca_full.explained_variance_
n_pc = int(np.argmax(np.cumsum(evr) >= 0.8) + 1)
n_kaiser = int((eig>1).sum())
print(f"PCA: cumulative variance {np.round(np.cumsum(evr)[:5],3)}; "
      f"80% rule -> {n_pc} PCs, Kaiser -> {n_kaiser} PCs")

#Parallel analysis: keep PCs whose eigenvalue beats random data
rand_eigs = np.array([PCA().fit(StandardScaler().fit_transform(
    RNG.normal(size=X.shape))).explained_variance_ for _ in range(500)])
pa_thresh = np.percentile(rand_eigs, 95, axis=0)
n_pa = int(np.argmax(eig < pa_thresh)) if (eig<pa_thresh).any() else len(eig)
print(f"Parallel analysis -> {n_pa} PCs")

loadings = pd.DataFrame(pca_full.components_[:4].T, index=FEATS, columns=[f"PC{i+1}" for i in range(4)])
print("\nLoadings (first 3PCs):\n",loadings.iloc[:,:3].round(2))
loadings.round(3).to_csv(OUT/"pca_loadings.csv")
Z = pca_full.transform(X)[:,:n_pc]
for i in range(3):
    D[f"PC{i+1}"] = pca_full.transform(X)[:,i]

#Clustering
L = linkage(Z,method="ward")
coph_r,_ = cophenet(L,pdist(Z))
sil = {k:silhouette_score(Z,fcluster(L,k,"maxclust")) for k in range(2,7)}
k_best = max(sil, key=sil.get)
print(f"\nCophenetic correlation (Ward):{coph_r:.3f}")
print("Silhouette by k:",{k:round(v,3) for k,v in sil.items()}, "->k=",k_best)
D["cluster"] = fcluster(L,k_best,"maxclust")

#Internal validation
def run_pipeline(Xs,k):
    Zs = PCA(n_components=n_pc).fit_transform(StandardScaler().fit_transform(Xs))
    return fcluster(linkage(Zs,"ward"),k,"maxclust")

#bootstrap stability
B = 500
jac = {c:[] for c in sorted(D.cluster.unique())}
raw = D[FEATS].values
for _ in range(B):
    idx = np.unique(RNG.choice(len(D),len(D),replace=True))
    lab_b = run_pipeline(raw[idx],k_best)
    orig = D.cluster.values[idx]
    for c in jac:
        A = set(np.where(orig==c)[0])
        if not A:
            continue
        jac[c].append(max(len(A & set(np.where(lab_b==b)[0]))/
                          len(A | set(np.where(lab_b == b)[0])) for b in np.unique(lab_b)))
stability = {int(c):round(float(np.mean(v)),3) for c,v in jac.items()}
print("Bootstrap mean Jaccard per cluster:",stability)

alt = {m:adjusted_rand_score(D.cluster,fcluster(linkage(Z,m),k_best,"maxclust")) for m in ["average","complete"]}
raw_ward = adjusted_rand_score(D.cluster, fcluster(linkage(X,"ward"),k_best,"maxclust"))
print("ARI vs alternative linkages:",{m:round(v,3) for m,v in alt.items()},
      f"| vs Ward on all standardised features (no PCA): {raw_ward:.3f}")

#external validation
ct = pd.crosstab(D.cluster, D.group)
print("\nCluster x diagnosis:\n",ct)
ari_dx = adjusted_rand_score(D.group,D.cluster)
fisher_p = stats.fisher_exact(ct.values)[1] if ct.shape == (2,2) else stats.chi2_contingency(ct)[1]
print(f"ARI with diagnosis: {ari_dx:.3f}; association p = {fisher_p:.4f}")

profile = D.groupby("cluster")[FEATS+["age"]].mean().round(3)
profile["n"] = D.cluster.value_counts().sort_index()
profile["pct_depressed"] = (D.groupby("cluster").group.apply(lambda g: (g == "Depressed").mean()*100).round(1))
profile["pct_female"] = (D.groupby("cluster").group.apply(lambda g: (g == "F").mean()*100).round(1))
print("\nCluster profiles:\n",profile.T)
profile.to_csv(OUT/"cluster_profiles.csv")

#MADRS within depressed participants
dep = D[D.group == "Depressed"]
madrs_by_c1 = dep.groupby("cluster").madrs.describe()[["count","mean","std"]]
print("\nMADRS by cluster (depressed only):\n",madrs_by_c1)
rho = {f"PC{i}":stats.spearmanr(dep[f"PC{i}"],dep.madrs) for i in (1,2,3)}
for k_,v in rho.items():
    print(f"Spearman {k_} vs MADRS (depressed): rho={v.statistic:.2f},p={v.pvalue:.3f}")

#diagnosis differences on PCs and confounds
pc_tests = {f"PC{i}":stats.mannwhitneyu(D.loc[D.group == 'Depressed', f"PC{i}"],
                                        D.loc[D.group == 'Control', f"PC{i}"])
            for i in (1,2,3)}

for k_,v in pc_tests.items():
    print(f"{k_} depressed vs control: Mann-Whitney p={v.pvalue:.4f}")

age_p = stats.kruskal(*[g.age for _,g in D.groupby("cluster")]).pvalue
sex_p = stats.chi2_contingency(pd.crosstab(D.cluster,D.sex))[1]
age_pc1 = stats.spearmanr(D.age,D.PC1)
print(f"Confounds: age across clusters p = {age_p:.3f}; sex p ={sex_p:.3f};"
      f"age vs PC1 rho={age_pc1.statistic:.2f} p={age_pc1.pvalue:.3f}")

#low-activity cluster is explained by hospitalisation rather than depression
extra = scores.set_index("number")[["inpatient","afftype"]]
D = D.join(extra)
dep = D[D.group == "Depressed"]
print("\nDepressed: cluster x setting (1=inpatient, 2=outpatient):\n",pd.crosstab(dep.cluster,dep.inpatient))
print("PC1 mean by setting (depressed):",dep.groupby("inpatient").PC1.mean().round(2).to_dict(),"p=",round(stats.mannwhitneyu(dep[dep.inpatient == 1].PC1,dep[dep.inpatient == 2].PC1).pvalue,4))
outp = D[(D.group == "Control") | (D.inpatient ==2)]
for pc in ("PC1","PC2"):
    d_,c_ = outp[outp.group == "Depressed"][pc], outp[outp.group == "Control"][pc]
    print(f"Outpatients vs controls {pc}:p={stats.mannwhitneyu(d_,c_).pvalue:.4f}")
for pc in ("PC1","PC2"):
    d_,c_ = D[D.group == "Depressed"][pc], D[D.group == "Control"][pc]
    auc = stats.mannwhitneyu(d_,c_).statistic / (len(d_)*len(c_))
    print(f"{pc} AUC depressed vs control: {auc:.2f}")
for k in (3,4):
    print(f"k={k} cluster x diagnosis",pd.crosstab(fcluster(L,k,"maxclust"),D.group).values.tolist())

#figures
pal = {1: '#1b9eff', 2:'#d95f02',3:"#7570b3",4:"#e7298a",5:"#66a61e", 6:"#e6ab02" }
fig,ax = plt.subplots(1,2,figsize=(9,3.4))
pcs = np.arange(1,len(evr)+1)
ax[0].plot(pcs,eig,"o-",label="Observed eigenvalue")
ax[0].plot(pcs,pa_thresh,"s--",c="grey",label="Parallel analysis (95th pct)")
ax[0].axhline(1,c="k",lw=6,ls=":",label="Kaiser = 1")
ax[0].set(xlabel = "Component",ylabel="Eigenvalue",title="Scree plot")
ax[0].legend(fontsize=7)
ax[1].bar(pcs,evr*100)
ax[1].plot(pcs,np.cumsum(evr)*100,"o-",c="C3")
ax[1].axhline(80,ls="--",c="grey")
ax[1].set(xlabel="Component",ylabel="% variance", title="Explained variance (cumulative in red)")
fig.tight_layout()
fig.savefig(OUT /"1_screen.png")
plt.close()

fig,ax = plt.subplots(figsize=(4.6,4.2))
im = ax.imshow(loadings.iloc[:,:3],cmap="RdBu_r",vmin= -0.7, vmax=0.7)
ax.set_xticks(range(3),[f"PC{i+1}\n({evr[i]*100:0.0f}%)" for i in range(3)])
ax.set_yticks(range(len(FEATS)),FEATS)
for i in range(len(FEATS)):
    for j in range(3):
        ax.text(j,i,f"{loadings.iloc[i,j]:.2f}",ha="center",va="center",fontsize=7)
fig.colorbar(im,shrink=0.8)
ax.set_title("PCA Loadings")
fig.tight_layout()
fig.savefig(OUT/"2_loadings.png")
plt.close()

fig,ax = plt.subplots(figsize=(10,4))
cut = (L[-(k_best - 1),2] + L[-k_best,2]) / 2
dn = dendrogram(L,labels=[("D" if g == "Depressed" else "C") + i.split("_")[1] for i,g in zip(D.index,D.group)],
                color_threshold=cut, above_threshold_color='grey',ax=ax,leaf_font_size=7)
for t in ax.get_xticklabels():
    t.set_color("#c0392b" if t.get_text().startswith("D") else "#2c3e50")
ax.axhline(cut,ls = "--", c="k",lw=0.7)
ax.set(title = f"Ward dendrogram on {n_pc} PC scores (D=depressed, red; C=control)",ylabel="Ward distance")
fig.tight_layout()
fig.savefig(OUT/"3_dendrogram.png")
plt.close()

fig,ax = plt.subplots(1,2,figsize=(9.5,3.8))
for c in sorted(D.cluster.unique()):
    for g,m in [("Depressed","^"),("Control","o")]:
        s = D[(D.cluster == c) & (D.group==g)]
        ax[0].scatter(s.PC1,s.PC2,c=pal[c],marker=m,s=38,edgecolor="k",lw=4,label=f"Cluster {c}-{g}")
ax[0].set(xlabel=f"PC1 ({evr[0]*100:.0f}%)",ylabel = f"PC2 ({evr[1]*100:.0f}%)", title="Participants in PC space")
ax[0].legend(fontsize=6.5)
ax[1].plot(list(sil),list(sil.values()),"o-")
ax[1].set(xlabel="Number of clusters k",ylabel="Mean Silhouette",title="Choosing k")
ax[1].axvline(k_best,ls="--",c="grey")
fig.tight_layout()
fig.savefig(OUT/"4_pc_scatter_silhouette.png")
plt.close()

zprof = pd.DataFrame(X,columns=FEATS,index=D.index).groupby(D.cluster).mean()
fig,ax = plt.subplots(figsize=(7,3.2))
for c in zprof.index:
    ax.plot(FEATS, zprof.loc[c], "o-", c=pal[c], label=f"Cluster {c} (n={profile.n[c]}, "
            f"{profile.pct_depressed[c]:.0f}% depressed)")
ax.axhline(0, c="k", lw=.5); ax.set_ylabel("Mean z-score")
ax.set_title("Cluster profiles on standardised features"); ax.legend(fontsize=7)
plt.xticks(rotation=35, ha="right"); fig.tight_layout()
fig.savefig(OUT / "5_cluster_profiles.png"); plt.close()
 
json.dump(dict(n=len(D), n_pc=n_pc, n_kaiser=n_kaiser, n_parallel=n_pa,
               evr=np.round(evr, 3).tolist(), cophenetic=round(coph_r, 3),
               silhouette={int(k): round(v, 3) for k, v in sil.items()}, k=k_best,
               bootstrap_jaccard=stability, ari_alt_linkage=alt, ari_no_pca=raw_ward,
               ari_diagnosis=ari_dx, association_p=fisher_p),
          open(OUT / "summary.json", "w"), indent=2, default=float)
print("\nSaved outputs to", OUT)