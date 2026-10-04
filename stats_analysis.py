import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sns.set_theme(style="whitegrid")
plt.rcParams['axes.titleweight'] = 'bold'

df = pd.read_csv("rainfall_clean.csv")
df['JJAS'] = df[['JUN','JUL','AUG','SEP']].sum(axis=1)
df['PERIOD'] = np.where(df['YEAR'] <= 1960, 'Early (1901-1960)', 'Recent (1961-2017)')

# =========================================================
# HYPOTHESIS 1 (PRIMARY): Welch's two-sample t-test
# H0: mean JJAS monsoon rainfall is equal between Early and Recent periods
# H1: mean JJAS monsoon rainfall differs between Early and Recent periods
# =========================================================
early = df.loc[df['PERIOD'] == 'Early (1901-1960)', 'JJAS']
recent = df.loc[df['PERIOD'] == 'Recent (1961-2017)', 'JJAS']

# Assumption checks
levene_stat, levene_p = stats.levene(early, recent)
shapiro_early = stats.shapiro(early.sample(min(500, len(early)), random_state=42))
shapiro_recent = stats.shapiro(recent.sample(min(500, len(recent)), random_state=42))

t_stat, t_p = stats.ttest_ind(early, recent, equal_var=False)  # Welch's t-test
mean_diff = recent.mean() - early.mean()
pooled_se = np.sqrt(early.var(ddof=1)/len(early) + recent.var(ddof=1)/len(recent))
ci_low, ci_high = mean_diff - 1.96*pooled_se, mean_diff + 1.96*pooled_se
cohens_d = mean_diff / np.sqrt(((len(early)-1)*early.var(ddof=1) + (len(recent)-1)*recent.var(ddof=1)) / (len(early)+len(recent)-2))

print("=== T-TEST: Early vs Recent JJAS rainfall ===")
print(f"n_early={len(early)}, n_recent={len(recent)}")
print(f"mean_early={early.mean():.2f}, mean_recent={recent.mean():.2f}, diff={mean_diff:.2f}")
print(f"Levene p={levene_p:.4f} (equal variance assumption)")
print(f"Shapiro-Wilk early p={shapiro_early.pvalue:.4g}, recent p={shapiro_recent.pvalue:.4g} (normality, informational only given large n)")
print(f"Welch t={t_stat:.3f}, p={t_p:.4g}")
print(f"95% CI of difference: [{ci_low:.2f}, {ci_high:.2f}]")
print(f"Cohen's d = {cohens_d:.3f}")

# =========================================================
# HYPOTHESIS 2 (SUPPORTING): Chi-square test of independence
# H0: Period (Early/Recent) and occurrence of an extreme monsoon (JJAS) total are independent
# H1: They are not independent
# =========================================================
thresh = df['JJAS'].quantile(0.90)
df['EXTREME_JJAS'] = df['JJAS'] > thresh

contingency = pd.crosstab(df['PERIOD'], df['EXTREME_JJAS'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
print("\n=== CHI-SQUARE: Period vs Extreme Monsoon Occurrence ===")
print(contingency)
print(f"chi2={chi2:.3f}, dof={dof}, p={chi_p:.4g}")
print("Expected frequencies:\n", np.round(expected,1))
# Cramer's V effect size
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape)-1)))
print(f"Cramer's V = {cramers_v:.3f}")

# =========================================================
# HYPOTHESIS 3 (SUPPORTING): One-way ANOVA
# H0: Mean JJAS monsoon rainfall is equal across the 5 selected climate-representative subdivisions
# H1: At least one subdivision's mean differs
# =========================================================
selected_subs = ['Konkan & Goa', 'West Rajasthan', 'Arunachal Pradesh', 'East Madhya Pradesh', 'Tamil Nadu']
anova_df = df[df['SUBDIVISION'].isin(selected_subs)]
groups = [anova_df.loc[anova_df['SUBDIVISION']==s, 'JJAS'] for s in selected_subs]
f_stat, anova_p = stats.f_oneway(*groups)
print("\n=== ONE-WAY ANOVA: JJAS rainfall across 5 representative subdivisions ===")
for s, g in zip(selected_subs, groups):
    print(f"{s}: n={len(g)}, mean={g.mean():.1f}, std={g.std():.1f}")
print(f"F={f_stat:.2f}, p={anova_p:.4g}")

tukey = pairwise_tukeyhsd(anova_df['JJAS'], anova_df['SUBDIVISION'], alpha=0.05)
print("\nTukey HSD post-hoc:\n", tukey)

# eta squared effect size
grand_mean = anova_df['JJAS'].mean()
ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)
ss_total = sum((anova_df['JJAS']-grand_mean)**2)
eta_sq = ss_between/ss_total
print(f"Eta squared = {eta_sq:.3f}")

# =========================================================
# VISUALIZATIONS
# =========================================================

# Viz 1: Overlaid histograms/KDE of JJAS rainfall, Early vs Recent
plt.figure(figsize=(10,6))
sns.histplot(early, color="#4a90d9", label="Early (1901-1960)", kde=True, stat="density", alpha=0.5, bins=30)
sns.histplot(recent, color="#d62728", label="Recent (1961-2017)", kde=True, stat="density", alpha=0.5, bins=30)
plt.axvline(early.mean(), color="#1f4e79", linestyle="--", linewidth=1.5)
plt.axvline(recent.mean(), color="#8b0000", linestyle="--", linewidth=1.5)
plt.title("Distribution of Monsoon (JJAS) Rainfall: Early vs Recent Period")
plt.xlabel("JJAS Rainfall (mm)"); plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("stat1_distribution_comparison.png", dpi=150)
plt.close()

# Viz 2: Boxplot with t-test annotation
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="PERIOD", y="JJAS", palette=["#4a90d9","#d62728"])
y_max = df['JJAS'].max()
plt.plot([0,0,1,1],[y_max*1.02, y_max*1.06, y_max*1.06, y_max*1.02], color='black')
sig = "p < 0.001" if t_p < 0.001 else f"p = {t_p:.3f}"
plt.text(0.5, y_max*1.08, f"Welch t = {t_stat:.2f}, {sig}", ha='center', fontsize=10)
plt.title("Monsoon Rainfall by Period (with Welch's t-test result)")
plt.ylabel("JJAS Rainfall (mm)"); plt.xlabel("")
plt.ylim(0, y_max*1.18)
plt.tight_layout()
plt.savefig("stat2_boxplot_ttest.png", dpi=150)
plt.close()

# Viz 3: Stacked bar for chi-square contingency (proportions)
prop = contingency.div(contingency.sum(axis=1), axis=0) * 100
plt.figure(figsize=(8,6))
prop.plot(kind="bar", stacked=True, color=["#4a90d9","#d62728"], ax=plt.gca())
plt.title(f"Proportion of Extreme Monsoon Years by Period (Chi-square p={chi_p:.3g})")
plt.ylabel("% of subdivision-years"); plt.xlabel("")
plt.legend(["Not Extreme", "Extreme (>90th pct)"], loc="upper right", bbox_to_anchor=(1.3,1))
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("stat3_chisquare_proportions.png", dpi=150)
plt.close()

# Viz 4: ANOVA boxplot across subdivisions
plt.figure(figsize=(11,6.5))
order = anova_df.groupby('SUBDIVISION')['JJAS'].mean().sort_values().index
sns.boxplot(data=anova_df, x="SUBDIVISION", y="JJAS", order=order, palette="viridis")
plt.title(f"Monsoon Rainfall Across Climate-Representative Subdivisions (ANOVA F={f_stat:.1f}, p<0.001)")
plt.ylabel("JJAS Rainfall (mm)"); plt.xlabel("")
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig("stat4_anova_boxplot.png", dpi=150)
plt.close()

# Viz 5: QQ-plots for normality check (early vs recent)
fig, axes = plt.subplots(1, 2, figsize=(11,5))
stats.probplot(early, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: Early Period JJAS Rainfall")
stats.probplot(recent, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Recent Period JJAS Rainfall")
plt.tight_layout()
plt.savefig("stat5_qqplots.png", dpi=150)
plt.close()

print("\nAll visualizations saved.")
print("\nThreshold (90th pct JJAS):", round(thresh,1))
