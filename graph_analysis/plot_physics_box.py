import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==============================================================================
# 0. STYLE CONFIGURATION FOR ACADEMIC PUBLICATION
# ==============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300
})

# ==============================================================================
# 1. PATHS & PARAMETERS
# ==============================================================================
ALGORITHMS = {
    "Expected SARSA": "resultados_E_SARSA",
    "SARSA": "resultados_SARSA",
    "Q-Learning": "resultados_TD"
}

SEED_PREFIXES = {
    "Expected SARSA": "E_SARSA_seed_",
    "SARSA": "SARSA_seed_",
    "Q-Learning": "TD_seed_"
}

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]
FINAL_EP = "6000"

# ==============================================================================
# 2. DATA LOADING FUNCTIONS
# ==============================================================================
def safe_load_csv(path):
    if not os.path.exists(path): return None
    for encoding in ["utf-8", "latin1", "windows-1252"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def process_telemetry(base_path, group_label, episode=None):
    t_csv_path = os.path.join(base_path, "trolebuses")

    if episode == 5999:
        t_csv_path = os.path.join(t_csv_path, f"ep_5999.csv")
    else:
        t_csv_path = os.path.join(t_csv_path, "ep_000.csv")

    # print(f"Loading telemetry data for {group_label} from {t_csv_path}")
    # input()
    df_t = safe_load_csv(t_csv_path)
    if df_t is not None:
        df_t.columns = df_t.columns.str.strip()
        df_t["Group"] = group_label
        df_t = df_t.sort_values(by=["tiempo_simulacion", "distancia_recorrida_m"], ascending=[True, False])
        # Calcular headway (Distance to Leader)
        df_t["dist_lider"] = df_t.groupby("tiempo_simulacion")["distancia_recorrida_m"].diff(-1).abs()
        
    return df_t

# ==============================================================================
# 3. DATA CONSOLIDATION
# ==============================================================================
print("Indexing folders and extracting operational telemetry...")
troles_list = []

for alg_name, dir_name in ALGORITHMS.items():
    prefix = SEED_PREFIXES[alg_name]
    
    for seed in SEEDS:
        seed_path = f"{dir_name}/{prefix}{seed}"
        
        # Load Episode 0 (Grouped as universal baseline)
        ep0_path = os.path.join(seed_path)
        t0 = process_telemetry(ep0_path, "Baseline (Ep. 1)", episode=0)
        if t0 is not None: troles_list.append(t0)
            
        # Load Converged Episode
        ep_final_path = os.path.join(seed_path)
        tf = process_telemetry(ep_final_path, f"{alg_name} (Ep. {FINAL_EP})", episode=5999)
        if tf is not None: troles_list.append(tf)

df_troles = pd.concat(troles_list, ignore_index=True) if troles_list else pd.DataFrame()

# Clean invalid distances
if not df_troles.empty:
    df_troles = df_troles.dropna(subset=["dist_lider"])
    df_troles = df_troles[(df_troles["dist_lider"] > 0) & (df_troles["dist_lider"] < 5000)]

# ==============================================================================
# 4. PLOTTING
# ==============================================================================
# fig, ax = plt.subplots(figsize=(6, 4.5))
fig, ax = plt.subplots(figsize=(9, 4.5))

# Brighter, modern hex colors
PALETTE = {
    "Baseline (Ep. 1)": "#95a5a6",          # Bright Grey
    f"Expected SARSA (Ep. {FINAL_EP})": "#00a8ff",  # Vibrant Blue
    f"SARSA (Ep. {FINAL_EP})": "#fbc531",           # Vibrant Yellow-Orange
    f"Q-Learning (Ep. {FINAL_EP})": "#4cd137"       # Vibrant Green
}
ORDER = list(PALETTE.keys())

if not df_troles.empty:
    sns.boxplot(
        data=df_troles, 
        x="Group", 
        y="dist_lider", 
        palette=PALETTE, 
        order=ORDER,
        ax=ax,
        width=0.55,
        # Applying transparency and crisp edges to avoid the "playdough" look
        boxprops=dict(alpha=0.65, edgecolor='#2f3640', linewidth=1),
        whiskerprops=dict(color='#2f3640', linewidth=0.8),
        capprops=dict(color='#2f3640', linewidth=0.8),
        medianprops=dict(color='#c23616', linewidth=0.8), # Red median for contrast
        flierprops=dict(marker="o", markersize=1.5, markerfacecolor='#7f8fa6', alpha=0.15, markeredgecolor='none')
    )
    
    # Ideal target line
    ax.axhline(y=1600.0, color="#2f3640", linestyle="--", linewidth=1, label="Target (1600 m)")
    
    # Titles and Labels
    # ax.set_title("Variance Compression in Headway Regimes", fontweight="bold", pad=7)
    ax.set_ylabel("Distance to Leader (m)", labelpad=7)
    ax.set_xlabel("")
    ax.set_ylim(350, 2550)
    ax.set_yticks([500, 1000, 1500, 2000, 2500])
    # Rotate X-axis text 90 degrees
    # plt.xticks(rotation=90)
    
    # Add subtle legend for the ideal target
    # ax.legend(loc="upper right", framealpha=0.7)

plt.tight_layout()
# bbox_inches="tight" ensures the 90-degree text doesn't get cut off when exporting to PDF

plt.savefig("headway_variance_compression.pdf", format='pdf', dpi=1200, bbox_inches="tight")
plt.savefig("headway_variance_compression.eps", format='eps', dpi=1200, bbox_inches="tight")
plt.savefig("headway_variance_compression.png", format='png', dpi=1200, bbox_inches="tight")
print(f"Plot successfully saved to headway_variance_compression.pdf")
plt.show()