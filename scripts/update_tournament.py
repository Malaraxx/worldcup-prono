"""
Pipeline de mise à jour résultats WC2026.

Usage :
  venv\\Scripts\\python scripts/update_tournament.py           # recalcule + propose commit/push
  venv\\Scripts\\python scripts/update_tournament.py --no-push # commit local uniquement
"""

import argparse
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED    = ROOT / "data" / "processed"
RAW          = ROOT / "data" / "raw"
RESULTS_PATH = RAW / "wc2026_results.csv"

logger = logging.getLogger(__name__)

_WC2026_FIRST_MATCH = datetime(2026, 6, 11, tzinfo=timezone.utc)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_results(path: Path, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Charge et valide le CSV de résultats.
    Lève SystemExit (code 1) si une erreur bloquante est détectée.
    """
    if not path.exists():
        logger.error("Fichier introuvable : %s", path)
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Impossible de lire %s : %s", path, exc)
        sys.exit(1)

    required = {"match_id", "home_score", "away_score"}
    missing  = required - set(df.columns)
    if missing:
        logger.error("Colonnes manquantes dans %s : %s", path.name, missing)
        sys.exit(1)

    if df.empty:
        return df.astype({"match_id": "Int64"})

    valid_ids = set(fixtures_df["match_id"].astype(int).tolist())
    errors    = []

    for i, row in df.iterrows():
        lineno = i + 2  # 0-indexed + header

        try:
            mid = int(row["match_id"])
        except (ValueError, TypeError):
            errors.append(f"Ligne {lineno} : match_id invalide ('{row['match_id']}')")
            continue

        if mid not in valid_ids:
            errors.append(f"Ligne {lineno} : match_id {mid} absent de fixtures.csv")
            continue

        for col in ("home_score", "away_score"):
            try:
                val = int(row[col])
                if val < 0:
                    errors.append(f"Ligne {lineno} : {col} < 0 pour match_id {mid}")
            except (ValueError, TypeError):
                errors.append(f"Ligne {lineno} : {col} invalide pour match_id {mid} ('{row[col]}')")

    dupes = df[df.duplicated(subset=["match_id"], keep=False)]["match_id"].dropna().unique()
    if len(dupes):
        errors.append(f"match_id en doublon : {sorted(int(x) for x in dupes)}")

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    df["match_id"]   = df["match_id"].astype(int)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


# ── Backup ────────────────────────────────────────────────────────────────────

def backup_matches_historical() -> Path:
    """Copie horodatée de matches_historical.csv dans le même dossier."""
    src = PROCESSED / "matches_historical.csv"
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = src.with_name(f"matches_historical.{ts}.bak")
    shutil.copy2(src, dst)
    logger.info("Backup : %s", dst.name)
    return dst


# ── Injection ─────────────────────────────────────────────────────────────────

def inject_results(
    results_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
) -> tuple[list[dict], int]:
    """
    Injecte les scores dans matches_historical.csv (idempotent).

    Retourne (report, n_changed) :
      report    — liste de dicts {match_id, home, away, score, status}
                  status : "injected" | "unchanged" | "skipped_ko"
      n_changed — nombre de lignes effectivement modifiées
    """
    if results_df.empty:
        return [], 0

    hist    = pd.read_csv(PROCESSED / "matches_historical.csv")
    fix_map = (
        fixtures_df.set_index("match_id")[["home_slot", "away_slot"]]
        .to_dict("index")
    )

    report    = []
    n_changed = 0

    for _, row in results_df.iterrows():
        mid = int(row["match_id"])
        hs  = int(row["home_score"])
        as_ = int(row["away_score"])

        fix_info = fix_map.get(mid)
        if not fix_info:
            logger.warning("match_id %d absent de fixtures_df", mid)
            continue

        home = fix_info["home_slot"]
        away = fix_info["away_slot"]

        mask = (
            (hist["home_team"] == home) &
            (hist["away_team"] == away) &
            (hist["tournament"] == "FIFA World Cup") &
            (hist["date"] >= "2026-06-01")
        )
        idx_list = hist.index[mask].tolist()

        if not idx_list:
            logger.warning(
                "match_id %d (%s vs %s) non trouvé dans matches_historical.csv "
                "(match KO ou noms non résolus — ignoré)",
                mid, home, away,
            )
            report.append({
                "match_id": mid, "home": home, "away": away,
                "score": f"{hs}-{as_}", "status": "skipped_ko",
            })
            continue

        idx = idx_list[0]
        cur_hs = hist.at[idx, "home_score"]
        cur_as = hist.at[idx, "away_score"]

        already_ok = (
            pd.notna(cur_hs) and pd.notna(cur_as)
            and int(float(cur_hs)) == hs
            and int(float(cur_as)) == as_
        )

        if already_ok:
            report.append({
                "match_id": mid, "home": home, "away": away,
                "score": f"{hs}-{as_}", "status": "unchanged",
            })
            continue

        hist.at[idx, "home_score"] = float(hs)
        hist.at[idx, "away_score"] = float(as_)
        n_changed += 1
        report.append({
            "match_id": mid, "home": home, "away": away,
            "score": f"{hs}-{as_}", "status": "injected",
        })

    if n_changed > 0:
        hist.to_csv(PROCESSED / "matches_historical.csv", index=False)

    return report, n_changed


# ── Pipeline recalcul ─────────────────────────────────────────────────────────

def _run_pipeline() -> dict:
    """Lance la chaîne complète et retourne les données avant/après pour le résumé."""
    from src.model.elo        import get_ratings_dict
    from src.model.predict    import run as run_predictions
    from src.strategy.optimal_pick import run as run_picks
    from src.simulation.monte_carlo import run_tournament

    # Snapshot Elo avant
    elo_path = PROCESSED / "elo_ratings.csv"
    if elo_path.exists():
        _df = pd.read_csv(elo_path)
        old_elo = dict(zip(_df["team"], _df["elo_rating"]))
    else:
        old_elo = {}

    # Snapshot picks avant
    picks_path = PROCESSED / "optimal_picks.csv"
    old_picks = pd.read_csv(picks_path) if picks_path.exists() else pd.DataFrame()

    logger.info("Recalcul Elo + prédictions...")
    run_predictions(recalculate_elo=True)

    logger.info("Recalcul picks optimaux...")
    new_picks = run_picks()

    logger.info("Simulation Monte-Carlo (10 000 itérations)...")
    run_tournament()

    # Snapshot Elo après (fichier réécrit par run_predictions)
    _df2 = pd.read_csv(elo_path)
    new_elo = dict(zip(_df2["team"], _df2["elo_rating"]))

    return {
        "old_elo":   old_elo,
        "new_elo":   new_elo,
        "old_picks": old_picks,
        "new_picks": new_picks,
    }


# ── Résumé ────────────────────────────────────────────────────────────────────

def _build_summary(inject_report: list[dict], pipeline_data: dict) -> str:
    lines = ["\n" + "=" * 62, "  RÉSUMÉ MISE À JOUR", "=" * 62]

    injected  = [r for r in inject_report if r["status"] == "injected"]
    unchanged = [r for r in inject_report if r["status"] == "unchanged"]
    skipped   = [r for r in inject_report if r["status"] == "skipped_ko"]

    lines.append(f"\n📥  Scores injectés : {len(injected)}")
    for r in injected:
        lines.append(f"    • #{r['match_id']:>3}  {r['home']} {r['score']} {r['away']}")
    if unchanged:
        lines.append(f"    (+ {len(unchanged)} déjà à jour, score inchangé)")
    if skipped:
        lines.append(f"    (+ {len(skipped)} ignoré(s) — match KO ou équipe inconnue)")

    # Variations Elo
    old_elo = pipeline_data["old_elo"]
    new_elo = pipeline_data["new_elo"]
    all_teams = set(old_elo) | set(new_elo)
    deltas = {
        t: new_elo.get(t, 1500.0) - old_elo.get(t, 1500.0)
        for t in all_teams
    }
    top5 = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    significant = [(t, d) for t, d in top5 if abs(d) > 0.05]

    if significant:
        lines.append("\n📊  Top 5 variations Elo :")
        for team, delta in significant:
            sign = "+" if delta >= 0 else ""
            lines.append(f"    • {team:<25} {sign}{delta:.1f}")
    else:
        lines.append("\n📊  Aucune variation Elo significative")

    # Picks modifiés
    old_picks = pipeline_data["old_picks"]
    new_picks = pipeline_data["new_picks"]
    if not old_picks.empty and not new_picks.empty and "mode_recommended" in old_picks.columns:
        merged = (
            old_picks[["match_id", "mode_recommended"]]
            .rename(columns={"mode_recommended": "mode_old"})
            .merge(
                new_picks[["match_id", "mode_recommended"]]
                .rename(columns={"mode_recommended": "mode_new"}),
                on="match_id", how="inner",
            )
        )
        changed = merged[merged["mode_old"] != merged["mode_new"]]
        if not changed.empty:
            lines.append(f"\n🔄  Picks avec mode modifié : {len(changed)}")
            for _, r in changed.head(5).iterrows():
                lines.append(
                    f"    • match_id {int(r['match_id']):>3} : "
                    f"{r['mode_old'].upper()} → {r['mode_new'].upper()}"
                )
        else:
            lines.append("\n✅  Aucun mode de pick modifié")

    lines.append("\n" + "=" * 62)
    return "\n".join(lines)


# ── Git ───────────────────────────────────────────────────────────────────────

def _git_commit_push(no_push: bool) -> None:
    """git add (CSVs ciblés) + commit + push optionnel."""
    today  = datetime.now(tz=timezone.utc).date()
    day_n  = (today - _WC2026_FIRST_MATCH.date()).days + 1
    msg    = f"chore: update results J+{day_n}"

    files_to_add = [
        "data/raw/wc2026_results.csv",
        "data/processed/matches_historical.csv",
        "data/processed/elo_ratings.csv",
        "data/processed/predictions.csv",
        "data/processed/optimal_picks.csv",
        "data/processed/score_distributions.csv",
        "data/processed/tournament_probabilities.csv",
        "data/processed/group_stage_simulations.csv",
        "data/processed/ko_predictions.csv",
    ]

    try:
        subprocess.run(["git", "add"] + files_to_add, cwd=ROOT, check=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=ROOT
        )
        if result.returncode == 0:
            print("ℹ️  Rien à commiter (aucun changement détecté par git).")
            return

        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        print(f"\n✅  Commit : '{msg}'")

        if no_push:
            print("ℹ️  --no-push : commit local créé, pas de push.")
        else:
            subprocess.run(["git", "push"], cwd=ROOT, check=True)
            print("✅  Push effectué → Streamlit Cloud se met à jour dans quelques minutes.")

    except subprocess.CalledProcessError as exc:
        logger.error("Erreur git : %s", exc)
        sys.exit(1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description="Mise à jour résultats WC2026")
    parser.add_argument(
        "--no-push", action="store_true",
        help="Crée le commit local sans pusher sur GitHub",
    )
    args = parser.parse_args()

    fixtures_df = pd.read_csv(PROCESSED / "fixtures.csv")
    results_df  = validate_results(RESULTS_PATH, fixtures_df)

    if results_df.empty:
        print(f"\nAucun résultat dans {RESULTS_PATH.name}. Rien à faire.")
        print("  → Remplis data/raw/wc2026_results.csv puis relance le script.")
        return

    print(f"\n{len(results_df)} résultat(s) trouvé(s) dans {RESULTS_PATH.name}")

    # Backup systématique avant toute écriture
    bak = backup_matches_historical()
    print(f"Backup créé : {bak.name}")

    # Injection
    inject_report, n_changed = inject_results(results_df, fixtures_df)
    if n_changed:
        print(f"{n_changed} score(s) injecté(s) dans matches_historical.csv")
    else:
        print("Scores déjà à jour — recalcul pipeline tout de même...")

    # Pipeline complet
    pipeline_data = _run_pipeline()

    # Résumé
    print(_build_summary(inject_report, pipeline_data))

    # Confirmation git
    response = input("Commiter + pusher sur GitHub ? [y/N] : ").strip().lower()
    if response == "y":
        _git_commit_push(no_push=args.no_push)
    else:
        print("Abandon git. CSV recalculés localement, non poussés.")


if __name__ == "__main__":
    main()
