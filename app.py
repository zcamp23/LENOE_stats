import os
import json
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

import base64
def show_crisp_image(img_path):
    """Εμφανίζει την εικόνα σε 100% πλήρη ανάλυση χωρίς το downsampling του st.image()."""
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<img src="data:image/png;base64,{b64_data}" style="width:100%; height:auto; border-radius:6px;">',
            unsafe_allow_html=True
        )

# -------------------------------------------------------------------------
# Ρύθμιση Σελίδας Streamlit
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="LENOE Dragster Stats (2021-2026)",
    page_icon="🏁",
    layout="wide"
)

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("outputs", "figures")
TABLES_DIR = os.path.join("outputs", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)


# Βοηθητική συνάρτηση περιγραφικών στατιστικών (από τα Notebooks 01, 02, 03)
def calc_stats(series):
    return pd.Series({
        'Count (Πλήθος)': series.count(),
        'Mean (Μέσος Όρος)': series.mean(),
        'Median (Διάμεσος)': series.median(),
        'Std (Τυπ. Απόκλιση)': series.std(),
        'Min (Ελάχιστο)': series.min(),
        'Max (Μέγιστο)': series.max()
    }).round(3)


# -------------------------------------------------------------------------
# Φόρτωση Δεδομένων:
# 1. Αν υπάρχουν τοπικά τα clean_*.csv, υπολογίζει τους πίνακες
#    με τον κώδικα των Notebooks και τους αποθηκεύει στο outputs/tables/.
# 2. Αν τρέχει στο Cloud (όπου τα clean_*.csv είναι gitignored), φορτώνει
#    απευθείας τους αποθηκευμένους πίνακες από το outputs/tables/!
# -------------------------------------------------------------------------
@st.cache_data
def load_dashboard_data():
    res_path = os.path.join(PROCESSED_DIR, "clean_results.csv")
    ent_path = os.path.join(PROCESSED_DIR, "clean_entries.csv")
    rec_path = os.path.join(PROCESSED_DIR, "clean_records.csv")

    if os.path.exists(res_path) and os.path.exists(ent_path) and os.path.exists(rec_path):
        results = pd.read_csv(res_path)
        entries = pd.read_csv(ent_path)
        records = pd.read_csv(rec_path)

        # --- KPIs ---
        kpis = {
            "total_passes": int(len(results)),
            "clean_passes": int((results['Is_Clean_Pass'] == True).sum()),
            "total_entries": int(len(entries)),
            "total_records": int(len(records)),
            "total_races": int(entries[entries['ΠΙΣΤΑ'] != 'ΔΟΚΙΜΕΣ']['Race_UID'].nunique())
        }
        with open(os.path.join(TABLES_DIR, "kpis.json"), "w", encoding="utf-8") as f:
            json.dump(kpis, f, ensure_ascii=False, indent=2)

        # =====================================================================
        # NOTEBOOK 01: Data Cleaning & Validation
        # =====================================================================
        entries_per_race = (
            entries[entries['ΠΙΣΤΑ'] != 'ΔΟΚΙΜΕΣ'].groupby(['Race_UID', 'ααΑγώνα', 'Ημερομηνία_Αγώνα', 'ΠΙΣΤΑ', 'Σχόλια'], dropna=False)['ααΣυμμετοχής']
            .count()
            .rename('Συμμετοχές (Entries)')
        )
        runs_per_race = results[results['ΠΙΣΤΑ'] != 'ΔΟΚΙΜΕΣ'].groupby('Race_UID')['ααΕπίδοσης'].count().rename('Σύνολο Περασμάτων (Results)')
        clean_runs_per_race = (
            results[results['Is_Clean_Pass'] == True]
            .groupby('Race_UID')['ααΕπίδοσης']
            .count()
            .rename('Καθαρά Περάσματα (Is_Clean_Pass)')
        )
        race_audit = (
            entries_per_race.reset_index()
            .merge(runs_per_race, on='Race_UID', how='left')
            .merge(clean_runs_per_race, on='Race_UID', how='left')
            .fillna({'Σύνολο Περασμάτων (Results)': 0, 'Καθαρά Περάσματα (Is_Clean_Pass)': 0})
        )
        race_audit['Σύνολο Περασμάτων (Results)'] = race_audit['Σύνολο Περασμάτων (Results)'].astype(int)
        race_audit['Καθαρά Περάσματα (Is_Clean_Pass)'] = race_audit['Καθαρά Περάσματα (Is_Clean_Pass)'].astype(int)

        validity_counts = (
            results.groupby(['Περιγραφή_valid'], dropna=False)['ααΕπίδοσης']
            .count()
            .reset_index(name='Πλήθος')
        )
        validity_counts['Ποσοστό (%)'] = (validity_counts['Πλήθος'] / len(results) * 100).round(2)
        validity_counts = validity_counts.sort_values(by='Πλήθος', ascending=False).reset_index(drop=True)

        raw_400 = results[(results['Απόσταση_Κατηγορίας_m'] == 400) & (results['Χρόνος400_sec'] > 0)]
        clean_400_nb1 = results[(results['Απόσταση_Κατηγορίας_m'] == 400) & (results['Is_Clean_Pass'] == True)]
        raw_200 = results[(results['Απόσταση_Κατηγορίας_m'] == 200) & (results['Χρόνος200_sec'] > 0)]
        clean_200_nb1 = results[(results['Απόσταση_Κατηγορίας_m'] == 200) & (results['Is_Clean_Pass'] == True)]

        def build_waterfall(df, dist):
            et_col = 'Χρόνος400_sec' if dist == 400 else 'Χρόνος200_sec'
            spd_col = 'Ταχύτητα400' if dist == 400 else 'Ταχύτητα200'
            et_bounds = (6.5, 20.0) if dist == 400 else (3.5, 14.0)
            spd_bounds = (80, 320) if dist == 400 else (60, 280)
            s0 = df
            s1 = s0[s0['ΠΙΣΤΑ'] != 'ΔΟΚΙΜΕΣ']
            s2 = s1[s1['Εγκυρη'] == 0]
            s3 = s2[s2['Χρόνος18_sec'].between(0.8, 3.5)]
            s4 = s3[s3[et_col].between(*et_bounds)]
            s5 = s4[s4[spd_col].between(*spd_bounds)]
            return [len(s0), len(s1), len(s2), len(s3), len(s4), len(s5)]

        wf_400 = build_waterfall(raw_400, 400)
        wf_200 = build_waterfall(raw_200, 200)

        waterfall_df = pd.DataFrame({
            'Βήμα Φιλτραρίσματος': [
                '0. Raw Περάσματα (Χρόνος Τερματισμού > 0s)',
                '1. Αφαίρεση Δοκιμών (ΠΙΣΤΑ != ΔΟΚΙΜΕΣ)',
                '2. Έγκυρα Αγωνιστικά Περάσματα (Εγκυρη == 0)',
                '3. Φίλτρο 18m / 60ft (0.8s έως 3.5s)',
                '4. Φίλτρο Χρόνου Τερματισμού (400m: 6.5-20s | 200m: 3.5-14s)',
                '5. Φίλτρο Ταχύτητας Εξόδου (400m: 80-320 | 200m: 60-280 km/h) [Clean]'
            ],
            'Υπόλοιπο 400m': wf_400,
            'Κόπηκαν 400m': [0] + list(np.diff(wf_400) * -1),
            'Υπόλοιπο 200m': wf_200,
            'Κόπηκαν 200m': [0] + list(np.diff(wf_200) * -1)
        })

        stats_comparison = pd.DataFrame({
            '400m ET Raw (s)': calc_stats(raw_400['Χρόνος400_sec']),
            '400m ET Clean (s)': calc_stats(clean_400_nb1['Χρόνος400_sec']),
            '200m ET Raw (s)': calc_stats(raw_200['Χρόνος200_sec']),
            '200m ET Clean (s)': calc_stats(clean_200_nb1['Χρόνος200_sec']),
            'Ταχύτητα 400m Raw (km/h)': calc_stats(raw_400['Ταχύτητα400']),
            'Ταχύτητα 400m Clean (km/h)': calc_stats(clean_400_nb1['Ταχύτητα400']),
            'Ταχύτητα 200m Raw (km/h)': calc_stats(raw_200['Ταχύτητα200']),
            'Ταχύτητα 200m Clean (km/h)': calc_stats(clean_200_nb1['Ταχύτητα200'])
        }).rename_axis('Στατιστικό').reset_index()

        # =====================================================================
        # NOTEBOOK 02: Historical Trends & Records (2003-2026)
        # =====================================================================
        rec_400 = records[
            (records['Απόσταση_Ρεκόρ_m'] == 400) &
            (~records['Περιγραφή'].astype(str).str.contains('<100', na=False)) &
            (records['Τύπος_Οχήματος_Ονομα'].isin(['Αυτοκίνητο', 'Μηχανή'])) &
            (records['Χρόνος400'] > 5.5)
        ].copy()

        rec_200 = records[
            (records['Απόσταση_Ρεκόρ_m'] == 200) &
            (records['Τύπος_Οχήματος_Ονομα'].isin(['Αυτοκίνητο', 'Μηχανή'])) &
            (records['Χρόνος400'] > 3.5)
        ].copy()

        cars_400_rec = rec_400[rec_400['Τύπος_Οχήματος_Ονομα'] == 'Αυτοκίνητο']
        bikes_400_rec = rec_400[rec_400['Τύπος_Οχήματος_Ονομα'] == 'Μηχανή']

        rec_stats_df = pd.DataFrame({
            'Αυτοκίνητο 400m ET (s)': calc_stats(cars_400_rec['Χρόνος400']),
            'Μηχανή 400m ET (s)': calc_stats(bikes_400_rec['Χρόνος400']),
            '200m (<200>) ET (s)': calc_stats(rec_200['Χρόνος400']),
            'Αυτοκίνητο Ταχ. 400m (km/h)': calc_stats(cars_400_rec[cars_400_rec['Ταχύτητα400'] > 80]['Ταχύτητα400']),
            'Μηχανή Ταχ. 400m (km/h)': calc_stats(bikes_400_rec[bikes_400_rec['Ταχύτητα400'] > 80]['Ταχύτητα400'])
        }).rename_axis('Στατιστικό').reset_index()

        cols_show = ['Ετος', 'Ονοματεπώνυμο', 'Οχημα', 'Περιγραφή', 'ΠΙΣΤΑ', 'Χρόνος400', 'Ταχύτητα400']
        top5_cars = cars_400_rec.nsmallest(5, 'Χρόνος400')[cols_show].reset_index(drop=True)
        top5_bikes = bikes_400_rec.nsmallest(5, 'Χρόνος400')[cols_show].reset_index(drop=True)

        track_summary = (
            rec_400.groupby('ΠΙΣΤΑ')
            .agg(
                Πλήθος_Ρεκόρ=('Id_Record', 'count'),
                Πρώτο_Έτος=('Ετος', 'min'),
                Τελευταίο_Έτος=('Ετος', 'max'),
                Ταχύτερο_ET_s=('Χρόνος400', 'min'),
                Μέσο_ET_s=('Χρόνος400', 'mean'),
                Διάμεσος_ET_s=('Χρόνος400', 'median'),
                Μέγιστη_Ταχύτητα_kmh=('Ταχύτητα400', 'max'),
                Μέση_Ταχύτητα_kmh=('Ταχύτητα400', 'mean')
            )
            .round(3)
            .sort_values(by='Πλήθος_Ρεκόρ', ascending=False)
            .reset_index()
        )

        trend_results = []
        for vtype, sub in [('Αυτοκίνητο', cars_400_rec), ('Μηχανή', bikes_400_rec)]:
            sub_spd = sub[sub['Ταχύτητα400'] > 80]
            r_et, p_et = stats.spearmanr(sub['Ετος'], sub['Χρόνος400'])
            r_spd, p_spd = stats.spearmanr(sub_spd['Ετος'], sub_spd['Ταχύτητα400'])
            trend_results.append({
                'Τύπος Οχήματος': vtype,
                'Spearman r (Έτος vs ET)': round(r_et, 3),
                'p-value (ET)': round(p_et, 4),
                'Spearman r (Έτος vs Ταχύτητα)': round(r_spd, 3),
                'p-value (Ταχύτητα)': round(p_spd, 4)
            })
        trend_df = pd.DataFrame(trend_results)

        heads_up_records = records[
            (records['Όριοχρόνου'] == 0) &
            (records['Περιγραφή'].notna()) &
            (~records['Περιγραφή'].str.contains('FREE PASS|Ελεύθερα|SHOW KART|TRUCK|100>', na=False)) &
            (records['Χρόνος400'] > 0)
        ].copy()
        et_idx = heads_up_records.groupby('Περιγραφή')['Χρόνος400'].idxmin()
        class_records_board = (
            heads_up_records.loc[et_idx]
            .sort_values(by=['Απόσταση_Ρεκόρ_m', 'Τύπος_Οχήματος_Ονομα', 'Χρόνος400'], ascending=[False, True, True])
            [['Περιγραφή', 'Τύπος_Οχήματος_Ονομα', 'Ονοματεπώνυμο', 'Χρόνος400', 'Ταχύτητα400', 'ΠΙΣΤΑ', 'Ημερομηνία', 'Οχημα']]
            .rename(columns={
                'Περιγραφή': 'Κατηγορία',
                'Τύπος_Οχήματος_Ονομα': 'Τύπος Οχήματος',
                'Χρόνος400': 'Χρόνος (s)',
                'Ταχύτητα400': 'Ταχύτητα (km/h)',
                'ΠΙΣΤΑ': 'Πίστα',
                'Οχημα': 'Όχημα'
            })
            .fillna('-')
            .reset_index(drop=True)
        )

        track_class_records = (
            class_records_board['Πίστα']
            .value_counts()
            .rename_axis('Πίστα')
            .reset_index(name='Πλήθος Ρεκόρ Κατηγοριών')
        )
        track_class_records['Ποσοστό (%)'] = (
            track_class_records['Πλήθος Ρεκόρ Κατηγοριών'] / len(class_records_board) * 100
        ).round(1)

        # =====================================================================
        # NOTEBOOK 03: Performance, Reaction Times & Track Analysis
        # =====================================================================
        results_nb3 = results[results['ΤύποςΟχήματος'] != 'Φορτηγό'].copy()
        clean = results_nb3[results_nb3['Is_Clean_Pass'] == True].copy()
        clean['Διαδρομή_Όνομα'] = clean['Διαδρομή'].map({1: 'Αριστερή (Lane 1)', 2: 'Δεξιά (Lane 2)'})
        clean['Φάση_Αγώνα'] = np.where(
            clean['ΠεριγραφήΚατάταξης'] == 'Με Συνολικό Χρόνο',
            'Eliminations (Κόντρες)',
            'Προκριματικά / Δοκιμαστικά'
        )
        clean_400 = clean[(clean['Απόσταση_Κατηγορίας_m'] == 400) & (clean['Χρόνος200_sec'] > 3.5)].copy()
        cars_400 = clean_400[clean_400['ΤύποςΟχήματος'] == 'Αυτοκίνητο']
        bikes_400 = clean_400[clean_400['ΤύποςΟχήματος'] == 'Μηχανή']

        splits_stats = pd.DataFrame({
            'Αυτοκίνητο 18m (s)': calc_stats(cars_400['Χρόνος18_sec']),
            'Μηχανή 18m (s)': calc_stats(bikes_400['Χρόνος18_sec']),
            'Αυτοκίνητο 200m (s)': calc_stats(cars_400['Χρόνος200_sec']),
            'Μηχανή 200m (s)': calc_stats(bikes_400['Χρόνος200_sec']),
            'Αυτοκίνητο BackHalf 200-400m (s)': calc_stats(cars_400['BackHalf_200_400_sec']),
            'Μηχανή BackHalf 200-400m (s)': calc_stats(bikes_400['BackHalf_200_400_sec']),
            'Αυτοκίνητο 400m ET (s)': calc_stats(cars_400['Χρόνος400_sec']),
            'Μηχανή 400m ET (s)': calc_stats(bikes_400['Χρόνος400_sec'])
        }).rename_axis('Στατιστικό').reset_index()

        reg_rows = []
        for vtype, sub in [('Αυτοκίνητο', cars_400), ('Μηχανή', bikes_400)]:
            slope_18, _, r_18, p_18, _ = stats.linregress(sub['Χρόνος18_sec'], sub['Χρόνος400_sec'])
            r_sp_18, _ = stats.spearmanr(sub['Χρόνος18_sec'], sub['Χρόνος400_sec'])
            _, _, r_200, _, _ = stats.linregress(sub['Χρόνος200_sec'], sub['Χρόνος400_sec'])
            reg_rows.append({
                'Τύπος Οχήματος': vtype,
                'Πλήθος (n)': len(sub),
                'Pearson r (18m vs 400m)': round(r_18, 3),
                'Spearman r (18m vs 400m)': round(r_sp_18, 3),
                'Κέρδος 400m ανά -0.1s στα 18m (s)': round(slope_18 * 0.1, 3),
                'Pearson r (200m vs 400m)': round(r_200, 3),
                'p-value (18m vs 400m)': f"{p_18:.2e}"
            })
        reg_df = pd.DataFrame(reg_rows)

        rt_clean = clean[clean['Net_RT_sec'].between(0.0, 1.5)].copy()
        rt_summary = (
            rt_clean.groupby(['Φάση_Αγώνα', 'ΤύποςΟχήματος'])['Net_RT_sec']
            .agg(Count='count', Mean='mean', Median='median', Std='std', Min='min', Max='max')
            .round(3)
            .reset_index()
        )

        champ_runs = results_nb3[(results_nb3['ΠΙΣΤΑ'] != 'ΔΟΚΙΜΕΣ') & (results_nb3['ααΣυμμετοχής'] > 0)].copy()
        champ_runs['Φάση_Αγώνα'] = np.where(
            champ_runs['ΠεριγραφήΚατάταξης'] == 'Με Συνολικό Χρόνο',
            'Eliminations (Κόντρες)',
            'Προκριματικά / Δοκιμαστικά'
        )
        rl_summary = champ_runs.groupby('Φάση_Αγώνα').agg(
            Σύνολο_Περασμάτων=('ααΕπίδοσης', 'count'),
            Red_Lights=('Εγκυρη', lambda s: (s == -1).sum())
        ).reset_index()
        rl_summary['Ποσοστό Red Light (%)'] = (
            rl_summary['Red_Lights'] / rl_summary['Σύνολο_Περασμάτων'] * 100
        ).round(2)

        lane_rows = []
        for vtype, sub in [('Αυτοκίνητο', cars_400), ('Μηχανή', bikes_400)]:
            l1 = sub[sub['Διαδρομή'] == 1]
            l2 = sub[sub['Διαδρομή'] == 2]
            _, p_18 = stats.mannwhitneyu(l1['Χρόνος18_sec'], l2['Χρόνος18_sec'])
            _, p_400 = stats.mannwhitneyu(l1['Χρόνος400_sec'], l2['Χρόνος400_sec'])
            lane_rows.append({
                'Τύπος Οχήματος': vtype,
                'Πλήθος (Αριστερή / Δεξιά)': f"{len(l1)} / {len(l2)}",
                'Διάμεσος 18m Αριστερή (s)': round(l1['Χρόνος18_sec'].median(), 3),
                'Διάμεσος 18m Δεξιά (s)': round(l2['Χρόνος18_sec'].median(), 3),
                'p-value 18m': round(p_18, 4),
                'Διάμεσος 400m Αριστερή (s)': round(l1['Χρόνος400_sec'].median(), 3),
                'Διάμεσος 400m Δεξιά (s)': round(l2['Χρόνος400_sec'].median(), 3),
                'p-value 400m': round(p_400, 4)
            })
        lane_df = pd.DataFrame(lane_rows)

        elim_runs = champ_runs[
            (champ_runs['Φάση_Αγώνα'] == 'Eliminations (Κόντρες)') &
            (champ_runs['Διαδρομή'].isin([1, 2]))
        ].copy()
        h2h_pairs = elim_runs.groupby(['Race_UID', 'ααΓύρου', 'ααΚόντρας']).filter(
            lambda g: len(g) == 2 and not g['Περιγραφή_valid'].isin(['Δεν προσήλθε', 'ΝΑΑγώνα', 'Αναγκαστ.Bye']).any()
        )
        h2h_wins = (
            h2h_pairs[h2h_pairs['ΑποτέλεσμαΚόντρας'] == 1]['Διαδρομή']
            .map({1: 'Αριστερή (Lane 1)', 2: 'Δεξιά (Lane 2)'})
            .value_counts()
            .reindex(['Αριστερή (Lane 1)', 'Δεξιά (Lane 2)'], fill_value=0)
            .rename_axis('Διαδρομή')
            .reset_index(name='Νίκες Head-to-Head')
        )
        total_races = len(h2h_pairs) // 2
        h2h_wins['Αναμενόμενες Νίκες (50%)'] = total_races / 2
        h2h_wins['Σύνολο Κοντρών'] = total_races
        h2h_wins['Ποσοστό Νικών (%)'] = (h2h_wins['Νίκες Head-to-Head'] / h2h_wins['Σύνολο Κοντρών'] * 100).round(1)
        chi2_stat, p_chi2 = stats.chisquare(f_obs=h2h_wins['Νίκες Head-to-Head'])
        h2h_wins['Chi-Square (χ²)'] = round(chi2_stat, 4)
        h2h_wins['p-value'] = round(p_chi2, 4)

        tables = {
            "race_audit": race_audit,
            "validity_counts": validity_counts,
            "waterfall_df": waterfall_df,
            "stats_comparison": stats_comparison,
            "rec_stats_df": rec_stats_df,
            "top5_cars": top5_cars,
            "top5_bikes": top5_bikes,
            "track_summary": track_summary,
            "trend_df": trend_df,
            "class_records_board": class_records_board,
            "track_class_records": track_class_records,
            "splits_stats": splits_stats,
            "reg_df": reg_df,
            "rt_summary": rt_summary,
            "rl_summary": rl_summary,
            "lane_df": lane_df,
            "h2h_wins": h2h_wins
        }

        # Αποθήκευση των συγκεντρωτικών πινάκων στο outputs/tables/
        for name, df in tables.items():
            df.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")

        return kpis, tables

    else:
        # Αν τρέχουμε στο Cloud (χωρίς τον φάκελο data/processed/), διαβάζουμε από το outputs/tables/
        with open(os.path.join(TABLES_DIR, "kpis.json"), "r", encoding="utf-8") as f:
            kpis = json.load(f)

        table_names = [
            "race_audit", "validity_counts", "waterfall_df", "stats_comparison",
            "rec_stats_df", "top5_cars", "top5_bikes", "track_summary", "trend_df",
            "class_records_board", "track_class_records", "splits_stats", "reg_df",
            "rt_summary", "rl_summary", "lane_df", "h2h_wins"
        ]
        tables = {}
        for name in table_names:
            tables[name] = pd.read_csv(os.path.join(TABLES_DIR, f"{name}.csv"))

        return kpis, tables


# Φόρτωση δεδομένων
kpis, tables = load_dashboard_data()

# -------------------------------------------------------------------------
# Κεντρικός Τίτλος & Καρτέλες (Tabs)
# -------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col2:
    st.image(os.path.join("docs", "lenoe2020.jpg")) 

st.title("Στατιστική Ανάλυση Πανελλήνιου Πρωταθλήματος Dragster (ΛΕΝΟΕ)")
st.markdown(
    "**Σύστημα εξαγωγής, καθαρισμού και στατιστικής ανάλυσης δεδομένων χρονομέτρησης "
    "(Περάσματα 2021–2026 & Ιστορικά Ρεκόρ 2003–2026)**"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Επισκόπηση & KPIs",
    "🗑️ 2. Καθαρισμός & Επικύρωση Δεδομένων",
    "🏆 3. Ιστορικά Ρεκόρ & Hall of Fame (2003–2026)",
    "⚡ 4. Ανάλυση Επιδόσεων, Αντίδρασης & Διαδρομών"
])

# =========================================================================
# TAB 1: Επισκόπηση & KPIs (Executive Dashboard)
# =========================================================================
with tab1:
    st.subheader("Συγκεντρωτικά Μεγέθη Επεξεργασμένων Δεδομένων")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Πρωταθληματικοί Αγώνες", kpis["total_races"])
    c2.metric("Συμμετοχές (Entries)", f"{kpis['total_entries']:,}".replace(",", "."))
    c3.metric("Σύνολο Περασμάτων (Raw)", f"{kpis['total_passes']:,}".replace(",", "."))
    c4.metric("Καθαρά Περάσματα (Clean)", f"{kpis['clean_passes']:,}".replace(",", "."))
    c5.metric("Ιστορικά Ρεκόρ (2003–2026)", f"{kpis['total_records']:,}".replace(",", "."))

    st.markdown("---")
    st.subheader("Σύνοψη Βασικών Ευρημάτων της Ανάλυσης")

    st.markdown("""
    - **1. Έλεγχος Ποιότητας & Καθαρισμός Χρονομέτρησης (`Notebook 01`):**
      Από το σύνολο των **14.096** καταγεγραμμένων περασμάτων, το **88.82% (12.520)** αποτελούν έγκυρα αγωνιστικά περάσματα. Μετά την εφαρμογή του πολυσταδιακού φίλτρου `Is_Clean_Pass` (αφαίρεση δοκιμών συστήματος, εγκαταλείψεων, Red Lights, Breakouts και σφαλμάτων φωτοκυττάρων), απομονώθηκαν **11.439 καθαρά περάσματα** (9.599 στα 400m και 1.840 στα 200m), μειώνοντας την τυπική απόκλιση του χρόνου 400m από **4.176 s** σε **1.940 s**.
    - **2. Διαχρονική Εξέλιξη Ρεκόρ 2003–2026 (`Notebook 02`):**
      Ο έλεγχος **Spearman Rank Correlation** αναδεικνύει ισχυρή, στατιστικά σημαντική πτωτική τάση στους χρόνους 400m των Μηχανών (**r = -0.290, p < 0.0001**) και ανοδική τάση στις ταχύτητες εξόδου (**r = +0.210, p < 0.0001**), με τον διάμεσο χρόνο ρεκόρ να υποχωρεί στα **9.38 s** το 2026. Στα Αυτοκίνητα καταγράφεται ήπια διαχρονική μείωση των χρόνων (**r = -0.119, p = 0.0006**).
    - **3. Επίδραση Εκκίνησης (18m / 60ft) στον Τελικό Χρόνο 400m (`Notebook 03`):**
      Ο χρόνος στα πρώτα 18m εξηγεί το **39.0%** της διακύμανσης του χρόνου 400m στα Αυτοκίνητα (**Pearson r = 0.624**) και το **20.5%** στις Μηχανές (**Pearson r = 0.453**). Κάθε **0.1 s** ταχύτερη εκκίνηση στα 18m συνδέεται κατά μέσο όρο με **0.349 s** ταχύτερο τερματισμό στα Αυτοκίνητα και **0.370 s** στις Μηχανές.
    - **4. Χρόνοι Αντίδρασης (`Net RT`) & Red Lights στις Κόντρες (`Notebook 03`):**
      Στις Κόντρες (Eliminations), ο διάμεσος καθαρός χρόνος αντίδρασης πέφτει κατά **58% στις Μηχανές** (από **0.357 s** σε **0.149 s**) και από **0.584 s** σε **0.464 s** στα Αυτοκίνητα, οδηγώντας σε σχεδόν **πενταπλασιασμό (4.7x) των Red Lights** (από **1.17%** στα Προκριματικά σε **5.50%** στις Κόντρες).
    - **5. Έλεγχος Ισοτιμίας Διαδρομών & Selection Bias (`Notebook 03`):**
      Παρότι η Δεξιά Διαδρομή (Lane 2) εμφανίζει υψηλότερο συνολικό ποσοστό νικών (**57.7% vs 42.3%**), ο έλεγχος **Wilcoxon** στους ίδιους οδηγούς αποδεικνύει ότι η πρόσφυση της πίστας είναι πρακτικά ταυτόσημη (**p = 0.211** στα Αυτοκίνητα, **p = 0.242** στις Μηχανές). Η διαφορά της περιόδου 2021–2024 οφειλόταν σε **Selection Bias** (τα φαβορί επέλεγαν τη δεξιά λωρίδα), ενώ τη διετία **2025–2026** οι νίκες ισορρόπησαν πλήρως σε **48.6% Αριστερή vs 51.4% Δεξιά** (**p = 0.5198**).
    """)

# =========================================================================
# TAB 2: Καθαρισμός & Επικύρωση Δεδομένων (Notebook 01)
# =========================================================================
with tab2:
    st.header("Notebook 01: Έλεγχος Ποιότητας & Καθαρισμός Δεδομένων")

    st.subheader("1. Έλεγχος Πληρότητας ανά Αγώνα (Συμμετοχές vs Περάσματα)")
    st.dataframe(tables["race_audit"], width='content', hide_index=True)

    st.subheader("2. Κατανομή Κωδικών Εγκυρότητας Περασμάτων (Valid)")
    st.dataframe(tables["validity_counts"], width='content', hide_index=True)

    st.subheader("3. Έλεγχος Διαρροής Φίλτρων (Waterfall Audit) & Σύγκριση Raw vs Clean")
    st.markdown(
        "**Σημείωση:** Περάσματα με μεμονωμένα σφάλματα φωτοκυττάρων (π.χ. μηδενική ταχύτητα εξόδου ή glitch στα 18m) "
        "φιλτράρονται εκτός από το `Is_Clean_Pass` ώστε να διασφαλιστεί η στατιστική ακεραιότητα των αναλύσεων."
    )
    st.dataframe(tables["waterfall_df"], width='content', hide_index=True)

    st.markdown("**Στατιστική Σύγκριση Πριν (Raw) vs Μετά τον Καθαρισμό (Clean):**")
    st.dataframe(tables["stats_comparison"], width='content', hide_index=True)

    st.subheader("4. Οπτικοποίηση Καθαρισμού Δεδομένων")
    show_crisp_image(os.path.join(FIGURES_DIR, "01_data_cleaning_validation.png"))

# =========================================================================
# TAB 3: Ιστορικά Ρεκόρ & Hall of Fame (Notebook 02)
# =========================================================================
with tab3:
    st.header("Notebook 02: Ιστορική Εξέλιξη Ατομικών Ρεκόρ & Ρεκόρ Κατηγοριών (2003–2026)")

    st.subheader("1. Περιγραφικά Στατιστικά Ατομικών Ρεκόρ ανά Τύπο Οχήματος")
    st.dataframe(tables["rec_stats_df"], width='content', hide_index=True)

    st.subheader("2. Πίνακας 'Hall of Fame' - Top 5 Ταχύτερα Ρεκόρ 400m Όλων των Εποχών")
    col_c, col_b = st.columns(2)
    with col_c:
        st.markdown("**Top 5 Ταχύτερα Ρεκόρ Χρόνου 400m - ΑΥΤΟΚΙΝΗΤΑ:**")
        st.dataframe(tables["top5_cars"], width='stretch', hide_index=True)
    with col_b:
        st.markdown("**Top 5 Ταχύτερα Ρεκόρ Χρόνου 400m - ΜΗΧΑΝΕΣ:**")
        st.dataframe(tables["top5_bikes"], width='stretch', hide_index=True)

    st.subheader("3. Ανάλυση Ατομικών Ρεκόρ Χρόνου 400m ανά Πίστα")
    st.dataframe(tables["track_summary"], width='content', hide_index=True)

    st.subheader("4. Στατιστικός Έλεγχος Διαχρονικής Τάσης (Spearman Rank Correlation)")
    st.dataframe(tables["trend_df"], width='content', hide_index=True)
    st.markdown("""
    **Τι δείχνει ο στατιστικός έλεγχος (Spearman r):**
    - **Στις Μηχανές:** Υπάρχει ισχυρή, στατιστικά σημαντική **πτωτική τάση** στους χρόνους 400m (**r = -0.290, p < 0.0001**) και **ανοδική τάση** στις ταχύτητες εξόδου (**r = +0.210, p < 0.0001**) από το 2004 έως το 2026 (ο διάμεσος χρόνος ρεκόρ έπεσε από τα ~10.3 s–11.0 s στα **9.38 s** το 2026).
    - **Στα Αυτοκίνητα:** Η συσχέτιση του έτους με την τελική ταχύτητα (**r = 0.031, p = 0.3747**) είναι πρακτικά μηδενική και στατιστικά μη σημαντική (p > 0.05). Η συσχέτιση του έτους με τον χρόνο τερματισμού (**r = -0.119, p = 0.0006**) είναι στατιστικά σημαντική αλλά ασθενής, δείχνοντας μια ήπια βελτίωση των χρόνων σε βάθος εικοσαετίας.
    """)

    st.subheader("5. Οπτικοποίηση Ιστορικών Τάσεων (2003–2026)")
    show_crisp_image(os.path.join(FIGURES_DIR, "02_historical_trends.png"))

    st.markdown("---")

    st.subheader("6. Πίνακας Απόλυτων Ρεκόρ ανά Κατηγορία (All-Time Class Records Board)")
    st.markdown("Περιλαμβάνει τις **26 αγωνιστικές Heads-Up κατηγορίες** χωρίς όριο χρόνου (`Όριοχρόνου == 0`).")
    col_board, col_track = st.columns([3, 1])
    with col_board:
        st.dataframe(tables["class_records_board"], width='stretch', hide_index=True)
    with col_track:
        st.markdown("**Κατοχή Ρεκόρ ανά Πίστα:**")
        st.dataframe(tables["track_class_records"], width='stretch', hide_index=True)

# =========================================================================
# TAB 4: Ανάλυση Επιδόσεων, Αντίδρασης & Διαδρομών (Notebook 03)
# =========================================================================
with tab4:
    st.header("Notebook 03: Στατιστική Ανάλυση Επιδόσεων (18m vs 400m), Χρόνων Αντίδρασης & Διαδρομών")

    st.subheader("1. Περιγραφικά Στατιστικά Ενδιάμεσων Χρόνων (Splits 18m, 200m, Back-Half, 400m)")
    st.dataframe(tables["splits_stats"], width='content', hide_index=True)

    st.subheader("2. Συσχέτιση & Γραμμική Παλινδρόμηση 18m (60ft) vs 400m ET")
    st.dataframe(tables["reg_df"], width='content', hide_index=True)
    st.markdown("""
    **Τι δείχνει ο έλεγχος συσχέτισης και γραμμικής παλινδρόμησης (18m & 200m vs 400m):**
    - **Αυτοκίνητα (n = 3.667):** Ο χρόνος στα πρώτα 18m (60ft) σχετίζεται ισχυρά και στατιστικά σημαντικά με τον τελικό χρόνο στα 400m (**Pearson r = 0.624, Spearman r = 0.608, p < 0.0001**). Ο συντελεστής προσδιορισμού (**R² = 0.390**) δείχνει ότι περίπου το 39% της διακύμανσης του χρόνου 400m συνδέεται με την επίδοση στα πρώτα 18 μέτρα. Από την κλίση της ευθείας παλινδρόμησης προκύπτει ότι **κάθε 0.1 s ταχύτερη εκκίνηση στα 18m συνδέεται κατά μέσο όρο με 0.349 s ταχύτερο τερματισμό στα 400m**.
    - **Μηχανές (n = 5.908):** Η συσχέτιση του χρόνου 18m με τον τελικό χρόνο 400m είναι μέτρια και στατιστικά σημαντική (**Pearson r = 0.453, Spearman r = 0.446, p < 0.0001, R² = 0.205**). **Κάθε 0.1 s ταχύτερη εκκίνηση στα 18m συνδέεται κατά μέσο όρο με 0.370 s ταχύτερο τερματισμό στα 400m**, με την εκκίνηση από μόνη της να εξηγεί μικρότερο ποσοστό της τελικής διακύμανσης στις μηχανές (~20.5%) σε σχέση με τα αυτοκίνητα (δηλ. Η εκκίνηση φαίνεται να προβλέπει λιγότερο τον τελικό χρόνο στις μηχανές).
    - **Σημείο των 200m:** Η συσχέτιση του ενδιάμεσου χρόνου 200m με τον τελικό χρόνο 400m είναι εξαιρετικά υψηλή τόσο στα αυτοκίνητα (**r = 0.923, R² = 0.852**) όσο και στις μηχανές (**r = 0.946, R² = 0.895**), επιβεβαιώνοντας ότι στα μισά της διαδρομής η τελική επίδοση έχει ήδη κριθεί σε ποσοστό 85%–90%.
    - **Προσοχή στην ερμηνεία:** Πρόκειται για στατιστική συσχέτιση μεταξύ όλων των κατηγοριών και όχι για απομονωμένη αιτιότητα. Ένα πιο ισχυρό και καλοστημένο όχημα που ξεκινά γρήγορα στα 18m συνεχίζει να επιταχύνει ταχύτερα και στην υπόλοιπη διαδρομή, επομένως μέρος του κέρδους οφείλεται στη συνολική δυναμικότητα του οχήματος και στο ότι οι κατηγορίες διαφέρουν σε ισχύ.
    """)

    st.subheader("3. Ανάλυση Καθαρών Χρόνων Αντίδρασης (Net_RT_sec) & Red Lights (Προκριματικά vs Eliminations)")
    st.markdown(
        "**Σημείωση:** `Καθαρός Χρόνος Αντίδρασης Net_RT_sec = ΧρόνοςΑντίδρασης_sec - 0.400`. Εξαιρέθηκαν περάσματα με καθαρό χρόνο αντίδρασης `Net RT > 1.5s`, καθώς αντιπροσωπεύουν μηχανική καθυστέρηση turbo / εμπλοκής DSG "
        "(spool-up / launch control) ή ηθελημένη αναμονή στα προκριματικά / ελεύθερα δοκιμαστικά, και όχι ανθρώπινο χρόνο αντίδρασης."
    )
    col_rt, col_rl = st.columns([3, 2])
    with col_rt:
        st.markdown("**Καθαρός Χρόνος Αντίδρασης (`Net_RT_sec`) ανά Φάση Αγώνα:**")
        st.dataframe(tables["rt_summary"], width='stretch', hide_index=True)
    with col_rl:
        st.markdown("**Ποσοστό Άκυρων Εκκινήσεων (`Red Lights`):**")
        st.dataframe(tables["rl_summary"], width='stretch', hide_index=True)

    st.markdown("""
    **Τι δείχνει η ανάλυση των Χρόνων Αντίδρασης & Red Lights:**
    - **Στις Μηχανές:** Ο διάμεσος καθαρός χρόνος αντίδρασης βελτιώνεται κατά **58%**, πέφτοντας από τα **0.357 s** στα Προκριματικά / Δοκιμαστικά στα **0.149 s** στις Κόντρες (Eliminations).
    - **Στα Αυτοκίνητα:** Ο διάμεσος καθαρός χρόνος αντίδρασης μειώνεται από τα **0.584 s** στα Προκριματικά / Δοκιμαστικά στα **0.464 s** στις Κόντρες.
    - **Άκυρες Εκκινήσεις - Red Lights:** Επειδή στις Κόντρες η νίκη κρίνεται στον συνολικό χρόνο (`Χρόνος Αντίδρασης + ET`), οι οδηγοί πιέζουν το φανάρι στο όριο, με αποτέλεσμα το ποσοστό των Red Lights σχεδόν να πενταπλασιάζεται (**4.7x**), ανεβαίνοντας από **1.17%** στα Προκριματικά σε **5.50%** στις Κόντρες!
    """)

    st.subheader("4. Έλεγχος Ισοτιμίας Διαδρομών (Lane Bias Audit: Αριστερή vs Δεξιά Διαδρομή)")
    st.markdown("**1. Σύγκριση Χρόνων 18m & 400m ανά Διαδρομή (Mann-Whitney U Test):**")
    st.dataframe(tables["lane_df"], width='content', hide_index=True)

    st.markdown("**2. Ποσοστό Νικών ανά Διαδρομή σε Πραγματικές Κόντρες (Head-to-Head Eliminations & Chi-Square Test):**")
    st.dataframe(tables["h2h_wins"], width='content', hide_index=True)

    st.markdown("""
    **Τι δείχνει ο στατιστικός έλεγχος Ισοτιμίας Διαδρομών (Mann-Whitney U / Chi-Square & Selection Bias):**
    - **Πρακτικά Ισοδύναμη Πρόσφυση (Διάμεσοι 18m & 400m):** Στην πράξη, η πρόσφυση των δύο διαδρομών είναι σχεδόν ταυτόσημη, καθώς οι διάμεσοι χρόνοι εκκίνησης στα πρώτα 18m διαφέρουν μόλις κατά **0.010 s στις Μηχανές** (1.602 s Αριστερή vs 1.592 s Δεξιά) και κατά **0.043 s στα Αυτοκίνητα** (1.816 s Αριστερή vs 1.773 s Δεξιά), ενώ αντίστοιχα μικρές είναι και οι διαφορές στους διαμέσους των 400m (9.598 s vs 9.547 s στις Μηχανές και 12.067 s vs 11.938 s στα Αυτοκίνητα).
    - **Το Φαινόμενο της Μεροληψίας Επιλογής (Selection Bias):** Το γεγονός ότι η Δεξιά Διαδρομή (Lane 2) εμφανίζει στατιστικά σημαντικά υψηλότερο συνολικό ποσοστό νικών στις κόντρες (**57.7% / 627 νίκες** έναντι **42.3% / 460 νίκες** της Αριστερής, χ² = 25.66, p < 0.0001) πιθανώς δεν οφείλεται σε κατασκευαστική ανισότητα της πίστας, αλλά σε **Selection Bias**: επειδή στους αγωνιζόμενους είχε επικρατήσει η αντίληψη ότι η δεξιά λωρίδα «κολλάει» καλύτερα, τα φαβορί -που ως ταχύτεροι των προκριματικών έχουν το δικαίωμα επιλογής λωρίδας στις κόντρες- επέλεγαν συστηματικά τη Δεξιά διαδρομή.
    - **Έλεγχος για τους ίδιους οδηγούς (Wilcoxon Signed-Rank Test):** Όταν συγκρίνονται οι ίδιοι οδηγοί στον ίδιο αγώνα και στις δύο λωρίδες, η διαφορά είναι αμελητέα (0.01–0.02 s) και **δεν είναι στατιστικά σημαντική** (Αυτοκίνητα: **p = 0.211**, Μηχανές: **p = 0.242**), επιβεβαιώνοντας ότι η διαφορά των συνολικών δεδομένων οφείλεται στο ποιοι οδηγοί τρέχουν σε κάθε λωρίδα και όχι στην πίστα.
    - **Εξισορρόπηση την Τελευταία Διετία (2025–2026 στο ~50/50):** Ενώ την περίοδο 2021–2024 η προτίμηση των φαβορί έδινε στη Δεξιά λωρίδα το 64.0% των νικών (348 vs 196), τα τελευταία 2 χρόνια (**2025–2026**) τα ποσοστά νικών σε 543 Head-to-Head κόντρες διαμορφώθηκαν σε **48.6% για την Αριστερή (264 νίκες)** και **51.4% για τη Δεξιά (279 νίκες)** - μια κατανομή πρακτικά **50/50** χωρίς καμία στατιστικά σημαντική διαφορά (**Chi-Square = 0.4144, p = 0.5198 > 0.05**)!
    """)

    st.subheader("5. Οπτικοποίηση Απόδοσης & Πίστας")
    show_crisp_image(os.path.join(FIGURES_DIR, "03_performance_and_track_analysis.png"))