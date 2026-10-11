#!/usr/bin/env python3
"""
Download ALL benchmark datasets (~5.7 GB) to data/ directory.

Auto-downloads: ETT, Weather, Traffic, Exchange, Illness, Solar, Tourism,
  M4, NAB, SMD, PSM, GIFT-Eval, UEA/UCR classification, M5
Requires manual registration: SWaT, Yahoo S5

Usage:
  python scripts/download_data.py              # download all
  python scripts/download_data.py --force      # re-download existing
"""
import os, sys, argparse, zipfile, shutil, tempfile
from urllib.request import urlopen, Request
import subprocess

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)
DOWNLOAD_FAILURES = []
FORCE = False


def record_failure(description, url, error):
    DOWNLOAD_FAILURES.append({"description": description, "url": url, "error": str(error)})

def fetch(url, dest, desc):
    """Download one permitted public source while preserving an existing file.

    This helper intentionally does not bypass registration, credential, or
    license gates imposed by upstream dataset providers.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not FORCE:
        print(f"  ✅ {desc}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  ⬇ {desc}...")
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=180) as r:
            with open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            raise RuntimeError("download produced an empty file")
        return True
    except Exception as e:
        if os.path.exists(dest) and os.path.getsize(dest) == 0:
            os.remove(dest)
        record_failure(desc, url, e)
        print(f"     ⚠️ {e}\n     URL: {url}")
        return False

def unzip(zip_path, to_dir):
    """Extract a downloaded archive while rejecting path traversal entries."""
    destination = os.path.abspath(to_dir)
    os.makedirs(destination, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = os.path.abspath(os.path.join(destination, member.filename))
            try:
                inside = os.path.commonpath([destination, target]) == destination
            except ValueError:
                inside = False
            if not inside:
                raise RuntimeError(f"archive entry escapes destination: {member.filename}")
            file_mode = (member.external_attr >> 16) & 0o170000
            if file_mode == 0o120000:
                raise RuntimeError(f"archive contains a symlink entry: {member.filename}")
        zf.extractall(destination)
    os.remove(zip_path)

def download_list(urls, dest_dir):
    """Download a list of (url, filename) pairs to dest_dir."""
    for url, fname in urls:
        fetch(url, os.path.join(dest_dir, fname), fname)

def main():
    """Retrieve supported sources; manual-only datasets remain manual."""
    global FORCE
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    FORCE = args.force

    print("=" * 60)
    print("ChronoScribe Dataset Downloader")

    # ═══════════════════════════════════════════════
    # 1. ETT (Electricity Transformer Temperature)
    print("\n[1/8] ETT-small")
    download_list([
        ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", "ETTh1.csv"),
        ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv", "ETTh2.csv"),
        ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv", "ETTm1.csv"),
        ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv", "ETTm2.csv"),
    ], os.path.join(DATA, "ETT-small"))

    # ═══════════════════════════════════════════════
    # 2. Standard multivariate datasets
    print("\n[2/8] Weather, Traffic, Exchange, Illness, Solar, Tourism")
    download_list([
        ("https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/weather/weather.csv", "weather.csv"),
    ], os.path.join(DATA, "weather"))
    fetch("https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt",
          os.path.join(DATA, "traffic", "traffic.csv"), "traffic")
    fetch("https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt",
          os.path.join(DATA, "exchange_rate", "exchange_rate.csv"), "exchange_rate")
    fetch("https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/data/illness/national_illness.csv",
          os.path.join(DATA, "illness", "national_illness.csv"), "illness")
    fetch("https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt",
          os.path.join(DATA, "solar", "solar_AL.txt"), "solar")

    # Tourism data (convert CSV to JSON format used by benchmark)
    tour_csv = os.path.join(DATA, "tourism", "tourism_data.csv")
    tour_json = os.path.join(DATA, "tourism", "tourism_data.json")
    if not os.path.exists(tour_json) or FORCE:
        fetch("https://raw.githubusercontent.com/robjhyndman/tourism/master/data/tourism.csv", tour_csv, "tourism CSV")
        if os.path.exists(tour_csv):
            import pandas as pd
            df = pd.read_csv(tour_csv)
            # Simple conversion: store as list of series
            data = []
            for col in df.columns[1:]:  # skip index
                data.append({"id": col, "values": df[col].dropna().tolist()})
            import json
            with open(tour_json, "w") as f:
                json.dump(data, f)
            print(f"  ✅ tourism_data.json converted")

    # Electricity (UCI zip → extract to electricity.csv)
    elec_dir = os.path.join(DATA, "electricity")
    elec_csv = os.path.join(elec_dir, "electricity.csv")
    if not os.path.exists(elec_csv) or FORCE:
        elec_zip = os.path.join(elec_dir, "electricity.zip")
        fetch("https://archive.ics.uci.edu/static/public/321/ld2011-2014.txt.zip", elec_zip, "electricity (UCI)")
        if os.path.exists(elec_zip) and os.path.getsize(elec_zip) > 0:
            unzip(elec_zip, elec_dir)
            # Rename LD2011_2014.txt to electricity.csv if needed
            for f in os.listdir(elec_dir):
                if f.endswith('.txt'):
                    os.rename(os.path.join(elec_dir, f), elec_csv)
                    print(f"  ✅ electricity.csv extracted")

    # ═══════════════════════════════════════════════
    # 3. M4 competition
    print("\n[3/8] M4 Competition")
    for freq in ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]:
        fetch(f"https://github.com/M4Competition/M4-methods/raw/master/Dataset/Train/{freq}-train.csv",
              os.path.join(DATA, "m4", f"{freq}-train.csv"), f"M4 {freq}")
    fetch("https://github.com/M4Competition/M4-methods/raw/master/Dataset/M4-info.csv",
          os.path.join(DATA, "m4", "M4-info.csv"), "M4 info")

    # ═══════════════════════════════════════════════
    # 4. NAB, SMD, PSM, GIFT-Eval
    print("\n[4/8] Anomaly detection (NAB, SMD, PSM) + GIFT-Eval")

    # NAB via GitHub zipball
    nab_dir = os.path.join(DATA, "nab")
    if not os.path.isdir(os.path.join(nab_dir, "realKnownCause")) or FORCE:
        print("  ⬇ NAB from GitHub...")
        try:
            import requests, tempfile
            r = requests.get("https://api.github.com/repos/numenta/NAB/zipball/master", stream=True, timeout=300)
            if r.status_code == 200:
                nab_zip = os.path.join(DATA, "nab.zip")
                with open(nab_zip, "wb") as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                with tempfile.TemporaryDirectory() as tmp:
                    unzip(nab_zip, tmp)
                    for root, dirs, files in os.walk(tmp):
                        if root.endswith("/data"):
                            for d in dirs:
                                src, dst = os.path.join(root, d), os.path.join(nab_dir, d)
                                if os.path.exists(dst): shutil.rmtree(dst)
                                shutil.move(src, dst)
                            break
                print("  ✅ NAB downloaded")
            else:
                record_failure("NAB", "https://api.github.com/repos/numenta/NAB/zipball/master", f"HTTP {r.status_code}")
        except Exception as e:
            record_failure("NAB", "https://api.github.com/repos/numenta/NAB/zipball/master", e)
            print(f"  ⚠️ {e}")

    # SMD from GitHub (OmniAnomaly repo)
    smd_dir = os.path.join(DATA, "SMD")
    if not os.path.isdir(smd_dir) or not os.listdir(smd_dir) or FORCE:
        print("  ⬇ SMD from GitHub...")
        try:
            with tempfile.TemporaryDirectory(prefix="chronoscribe-omni-smd-") as clone_dir:
                clone = subprocess.run(
                    ["git", "clone", "--depth=1", "--filter=blob:none",
                     "https://github.com/NetManAIOps/OmniAnomaly.git", clone_dir],
                    capture_output=True, timeout=120, check=False,
                )
                if clone.returncode != 0:
                    detail = clone.stderr.decode(errors="replace").strip() or f"git exited {clone.returncode}"
                    raise RuntimeError(detail)
                smd_src = os.path.join(clone_dir, "ServerMachineDataset")
                if os.path.isdir(smd_src):
                    for sub in ["train", "test", "test_label"]:
                        src = os.path.join(smd_src, sub)
                        dst = os.path.join(smd_dir, sub)
                        if os.path.isdir(src):
                            if os.path.exists(dst): shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                    print("  ✅ SMD downloaded (raw format)")
                else:
                    raise RuntimeError("cloned repository has no ServerMachineDataset directory")
        except Exception as e:
            record_failure("SMD", "https://github.com/NetManAIOps/OmniAnomaly.git", e)
            print(f"  ⚠️ {e}")
            print("  Manual: git clone https://github.com/NetManAIOps/OmniAnomaly.git")

    # PSM from GitHub
    print("  ⬇ PSM from GitHub...")
    for f in ["train.csv", "test.csv"]:
        fetch(f"https://raw.githubusercontent.com/eBay/RANSynCoders/main/data/PSM/{f}",
              os.path.join(DATA, "PSM", f), f"PSM {f}")
    fetch("https://raw.githubusercontent.com/eBay/RANSynCoders/main/data/PSM/test_label.csv",
          os.path.join(DATA, "PSM", "test_label.csv"), "PSM test_label")

    # GIFT-Eval from HuggingFace
    gift_dir = os.path.join(DATA, "GIFT-Eval")
    if not os.path.isdir(gift_dir) or not os.listdir(gift_dir) or FORCE:
        print("  ⬇ GIFT-Eval from HuggingFace...")
        try:
            from datasets import load_dataset
            load_dataset("Salesforce/GiftEval", cache_dir=gift_dir)
            print("  ✅ GIFT-Eval downloaded")
        except Exception as e:
            record_failure("GIFT-Eval", "https://huggingface.co/datasets/Salesforce/GiftEval", e)
            print(f"  ⚠️ {e}")

    # ═══════════════════════════════════════════════
    # 5. M5 (via kagglehub)
    print("\n[5/8] M5 (Kaggle)")
    m5_dir = os.path.join(DATA, "m5-forecasting-accuracy")
    if not os.path.exists(os.path.join(m5_dir, "sales_train_evaluation.csv")) or FORCE:
        try:
            import kagglehub
            path = kagglehub.dataset_download("competitions/m5-forecasting-accuracy")
            for f in os.listdir(path):
                src = os.path.join(path, f)
                dst = os.path.join(m5_dir, f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            print("  ✅ M5 downloaded via kagglehub")
        except ImportError:
            record_failure("M5", "kaggle://competitions/m5-forecasting-accuracy", "kagglehub is not installed")
            print("  ⚠️ pip install kagglehub && kagglehub.login()")
            print("     Then re-run this script.")
        except Exception as e:
            record_failure("M5", "kaggle://competitions/m5-forecasting-accuracy", e)
            print(f"  ⚠️ {e}")

    # ═══════════════════════════════════════════════
    # 6. UEA/UCR classification (.ts files)
    print("\n[6/8] UEA/UCR Classification")
    uea_ucr = [
        "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat",
        "JapaneseVowels", "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2",
        "SpokenArabicDigits", "UWaveGestureLibrary",
    ]
    for name in uea_ucr:
        dest = os.path.join(DATA, name)
        fetch(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}_TRAIN.ts",
              os.path.join(dest, f"{name}_TRAIN.ts"), f"{name} TRAIN")
        fetch(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}_TEST.ts",
              os.path.join(dest, f"{name}_TEST.ts"), f"{name} TEST")

    # MONSTER and UCR sub-datasets
    print("  ⬇ MONSTER/UCR from timeseriesclassification.com...")
    monster_list = ["MonsterActivity", "MonsterECG", "MonsterTraffic"]
    ucr_list = ["Cricket", "ECG200", "FordA", "FordB", "HandOutlines",
                "StarLightCurves", "Strawberry", "Wafer"]
    for name in monster_list:
        dest_monster = os.path.join(DATA, "monster_classification")
        fetch(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}_TRAIN.ts",
              os.path.join(dest_monster, f"{name}_TRAIN.ts"), f"MONSTER {name} TRAIN")
        fetch(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}_TEST.ts",
              os.path.join(dest_monster, f"{name}_TEST.ts"), f"MONSTER {name} TEST")
    for name in ucr_list:
        dest_ucr = os.path.join(DATA, "ucr_classification")
        fetch(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}_TRAIN.ts",
              os.path.join(dest_ucr, f"{name}_TRAIN.ts"), f"UCR {name} TRAIN")
        fetch(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}_TEST.ts",
              os.path.join(dest_ucr, f"{name}_TEST.ts"), f"UCR {name} TEST")

    # ═══════════════════════════════════════════════
    # 7. Manual-only datasets
    print("\n[7/8] Manual registration datasets")
    print("""
  SWaT (iTrust, Singapore University):
    → Register at https://itrust.sutd.edu.sg
    → Place .xlsx files in data/SWaT/

  Yahoo S5 (Yahoo Webscope):
    → Register at https://webscope.sandbox.yahoo.com
    → Place benchmark folders (A1-A4, artificialWithAnomaly) in data/yahoo_s5/
""")

    # ═══════════════════════════════════════════════
    # 8. Summary
    print("[8/8] Done!")
    print("=" * 60)
    print("Auto-downloaded: ETT, Weather, Traffic, Exchange, Illness, Solar,")
    print("  Tourism, M4, NAB, SMD, PSM, GIFT-Eval, M5, UEA/UCR (26 datasets)")
    print("Manual: SWaT, Yahoo S5 (2 datasets — require institutional registration)")
    print("=" * 60)
    if DOWNLOAD_FAILURES:
        print("Download incomplete; failed sources:")
        for failure in DOWNLOAD_FAILURES:
            print(f"  - {failure['description']}: {failure['error']} ({failure['url']})")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
