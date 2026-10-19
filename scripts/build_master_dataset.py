import os
import glob
import pandas as pd
import numpy as np
from access_parser import AccessParser

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_tables_from_mdb(mdb_path):
    """Διαβάζει τους πίνακες από ένα αρχείο Access (.mdb) χωρίς ODBC driver."""
    db = AccessParser(mdb_path)
    table_names = ['Results', 'Entries', 'Sessions', 'Drivers', 'Races', 'Tracks', 'Classes', 'Valid', 'Records']
    tables = {}
    for t in table_names:
        raw_table = db.parse_table(t)
        df = pd.DataFrame(raw_table)
        
        # Εξασφάλιση ομοιομορφίας στο όνομα της στήλης πίστας (αν το .mdb την έχει ως 'Πίστα' ή 'αα_Πίστα')
        if t in ['Races', 'Records'] and 'αα_Πίστας' not in df.columns:
            if 'αα_Πίστα' in df.columns:
                df = df.rename(columns={'αα_Πίστα': 'αα_Πίστας'})
            elif 'Πίστα' in df.columns:
                df = df.rename(columns={'Πίστα': 'αα_Πίστας'})
                
        tables[t] = df
    return tables


def load_tables_from_xlsx(folder_path):
    """Fallback: Διαβάζει τους πίνακες από τα εξαγόμενα .xlsx αρχεία."""
    tables = {}
    table_names = ['Results', 'Entries', 'Sessions', 'Drivers', 'Races', 'Tracks', 'Classes', 'Valid', 'Records']
    for t in table_names:
        file_path = os.path.join(folder_path, f"{t}.xlsx")
        if os.path.exists(file_path):
            xls = pd.ExcelFile(file_path)
            tables[t] = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])
        else:
            raise FileNotFoundError(f"Λείπει το αρχείο: {file_path}")
    return tables


def process_single_database(tables, db_label):
    """Ενώνει και καθαρίζει τους πίνακες μιας αγωνιστικής σεζόν."""
    results = tables['Results'].copy()
    entries = tables['Entries'].copy()
    sessions = tables['Sessions'].copy()
    drivers = tables['Drivers'].copy()
    races = tables['Races'].copy()
    tracks = tables['Tracks'].copy()
    classes = tables['Classes'].copy()
    valid = tables['Valid'].copy()

    # Καθαρισμός ονομάτων οδηγών
    drivers['Ονοματεπώνυμο'] = (
        drivers['Επώνυμο'].fillna('').astype(str).str.strip() + ' ' +
        drivers['Ονομα'].fillna('').astype(str).str.strip()
    ).str.strip()

    # Ένωση Races με Tracks
    races_tracks = races.merge(tracks, on='αα_Πίστας', how='left')
    races_tracks['Ημερομηνία_Αγώνα'] = pd.to_datetime(races_tracks['Ημερομηνία'], errors='coerce')
    races_tracks['Ετος_Αγώνα'] = races_tracks['Ημερομηνία_Αγώνα'].dt.year.fillna(0).astype(int)
    races_tracks['Race_UID'] = db_label + "_R" + races_tracks['ααΑγώνα'].astype(str)

    # 1. Καθαρισμός Πίνακα Συμμετοχών (Entries)
    clean_entries = (
        entries
        .merge(drivers[['ααΟδηγού', 'Ονοματεπώνυμο']], left_on='αρΟδηγού', right_on='ααΟδηγού', how='left')
        .merge(races_tracks[['ααΑγώνα', 'Race_UID', 'Ημερομηνία_Αγώνα', 'ΠΙΣΤΑ', 'Σχόλια']], on='ααΑγώνα', how='left')
    )
    clean_entries['DB_Source'] = db_label

    # 2. Ένωση Results με Sessions, Entries, Drivers, Races, Tracks & Valid
    df = (
        results
        .merge(sessions, on='ααΓύρου', how='left', suffixes=('', '_session'))
        .merge(entries, on='ααΣυμμετοχής', how='left', suffixes=('', '_entry'))
        .merge(drivers[['ααΟδηγού', 'Ονοματεπώνυμο']], left_on='αρΟδηγού', right_on='ααΟδηγού', how='left')
        .merge(races_tracks[['ααΑγώνα', 'Race_UID', 'Ημερομηνία_Αγώνα', 'ΠΙΣΤΑ', 'Σχόλια']], 
               left_on='ααΑγώνα', right_on='ααΑγώνα', how='left')
        .merge(valid, left_on='Εγκυρη', right_on='Κωδ_Εγκυρότητας', how='left', suffixes=('', '_valid'))
    )
    df['DB_Source'] = db_label

    # 3. Μετατροπή Χρόνων από Milliseconds (ms) σε Δευτερόλεπτα (s)
    time_cols = [
        'ΧρόνοςΑντίδρασης', 'Χρόνος18', 'Χρόνος100', 'Χρόνος190',
        'Χρόνος200', 'Χρόνος290', 'Χρόνος300', 'Χρόνος390', 'Χρόνος400'
    ]
    for col in time_cols:
        if col in df.columns:
            df[f"{col}_sec"] = df[col] / 1000.0

    # 4. Feature Engineering για Στατιστική Ανάλυση
    # Καθαρό Reaction Time πάνω στο Pro Tree (0.400s = 0.000s)
    df['Net_RT_sec'] = df['ΧρόνοςΑντίδρασης_sec'] - 0.400

    # Απόσταση Τερματισμού Κατηγορίας (400m vs 200m vs 100m vs 18m)
    cat_col = df['ΠεριγραφήΚατηγορίας'].fillna(df['ΠεριγραφήΚατηγορίας_entry']).fillna('')
    df['Απόσταση_Κατηγορίας_m'] = np.select(
        [
            cat_col.str.contains('<200', na=False),
            cat_col.str.contains('<300', na=False),
            cat_col.str.contains('<100', na=False),
            cat_col.str.contains('<18>', na=False)
        ],
        [200, 300, 100, 18],
        default=400
    )

    # Back-Half (200m -> 400m) χρόνος και κέρδος ταχύτητας (μόνο για έγκυρα 400m περάσματα)
    valid_400_mask = (df['Χρόνος400_sec'] > 0) & (df['Χρόνος200_sec'] > 0)
    df['BackHalf_200_400_sec'] = np.where(valid_400_mask, df['Χρόνος400_sec'] - df['Χρόνος200_sec'], np.nan)
    df['Speed_Gain_200_400_kmh'] = np.where(
        (df['Ταχύτητα400'] > 0) & (df['Ταχύτητα200'] > 0),
        df['Ταχύτητα400'] - df['Ταχύτητα200'],
        np.nan
    )

    # Flag για Καθαρό Αγωνιστικό Πέρασμα (χωρίς ΔΟΚΙΜΕΣ, εγκαταλείψεις ή σφάλματα φωτοκυττάρων)
    df['Is_Clean_Pass'] = (
        (df['ΠΙΣΤΑ'] != 'ΔΟΚΙΜΕΣ') &
        (df['Εγκυρη'] == 0) &
        (df['Χρόνος18_sec'].between(0.8, 3.5)) &
        (
            ((df['Απόσταση_Κατηγορίας_m'] == 400) & df['Χρόνος400_sec'].between(5.5, 20.0) & (df['Ταχύτητα400'] > 80)) |
            ((df['Απόσταση_Κατηγορίας_m'] == 200) & df['Χρόνος200_sec'].between(3.5, 14.0) & (df['Ταχύτητα200'] > 60))
        )
    )

    return df, clean_entries


def process_records_table(tables):
    """Καθαρίζει και εμπλουτίζει τον ιστορικό πίνακα Records (2003-2026)."""
    records = tables['Records'].copy()
    tracks = tables['Tracks'].copy()
    classes = tables['Classes'].copy()
    drivers = tables['Drivers'].copy()

    drivers['Ονοματεπώνυμο'] = (
        drivers['Επώνυμο'].fillna('').astype(str).str.strip() + ' ' +
        drivers['Ονομα'].fillna('').astype(str).str.strip()
    ).str.strip()

    rec_df = (
        records
        .merge(tracks, on='αα_Πίστας', how='left')
        .merge(classes, on='ααΚατηγορίας', how='left')
        .merge(drivers[['ααΟδηγού', 'Ονοματεπώνυμο']], left_on='αρΟδηγού', right_on='ααΟδηγού', how='left')
    )

    rec_df['Ημερομηνία'] = pd.to_datetime(rec_df['Ημερομηνία'], errors='coerce')
    rec_df['Ετος'] = rec_df['Ημερομηνία'].dt.year

    # Τύπος Οχήματος (1=Αυτοκίνητο, 2=Μηχανή, 3=Τρακτέρ/Truck)
    rec_df['Τύπος_Οχήματος_Ονομα'] = rec_df['Τύπος'].map({1: 'Αυτοκίνητο', 2: 'Μηχανή', 3: 'Truck'}).fillna('Άλλο')

    # Απόσταση ρεκόρ (400m vs 200m)
    rec_df['Απόσταση_Ρεκόρ_m'] = np.where(
        rec_df['Περιγραφή'].astype(str).str.contains('<200', na=False), 200, 400
    )

    return rec_df


def main():
    mdb_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.mdb")))
    all_results = []
    all_entries = []
    latest_tables = None

    if mdb_files:
        print(f"Βρέθηκαν {len(mdb_files)} αρχεία .mdb στο {RAW_DIR}.")
        for mdb_path in mdb_files:
            db_label = os.path.splitext(os.path.basename(mdb_path))[0]
            print(f" -> Επεξεργασία βάσης: {db_label}...")
            tables = load_tables_from_mdb(mdb_path)
            res_df, ent_df = process_single_database(tables, db_label)
            all_results.append(res_df)
            all_entries.append(ent_df)
            latest_tables = tables
    else:
        print(f"Δεν βρέθηκαν .mdb. Ανάγνωση των εξαγόμενων .xlsx αρχείων από το {RAW_DIR}...")
        tables = load_tables_from_xlsx(RAW_DIR)
        res_df, ent_df = process_single_database(tables, "Season_Current")
        all_results.append(res_df)
        all_entries.append(ent_df)
        latest_tables = tables

    master_results = pd.concat(all_results, ignore_index=True)
    master_entries = pd.concat(all_entries, ignore_index=True)
    master_records = process_records_table(latest_tables) # Δεν χρειάζεται να ενωθεί από πολλές βάσεις, η πιο πρόσφατη βάση περιέχει όλο το ιστορικό ρεκόρ (2003–2026).

    # Αποθήκευση σε UTF-8 CSV με BOM (utf-8-sig) για τέλεια εμφάνιση ελληνικών παντού
    res_out = os.path.join(PROCESSED_DIR, "clean_results.csv")
    ent_out = os.path.join(PROCESSED_DIR, "clean_entries.csv")
    rec_out = os.path.join(PROCESSED_DIR, "clean_records.csv")

    master_results.to_csv(res_out, index=False, encoding="utf-8-sig")
    master_entries.to_csv(ent_out, index=False, encoding="utf-8-sig")
    master_records.to_csv(rec_out, index=False, encoding="utf-8-sig")

    print("\nΟλοκληρώθηκε επιτυχώς η δημιουργία των καθαρών αρχείων!")
    print(f"1. {res_out}: {len(master_results)} περάσματα ({master_results['Is_Clean_Pass'].sum()} καθαρά αγωνιστικά)")
    print(f"2. {ent_out}: {len(master_entries)} συμμετοχές")
    print(f"3. {rec_out}: {len(master_records)} ιστορικά ρεκόρ")


if __name__ == "__main__":
    main()