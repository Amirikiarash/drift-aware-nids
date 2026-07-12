"""Fetch the UGR'16 v1 feature CSVs (real ISP traffic + labelled attacks) used by
Step 3, from the public josecamachop/UGR16_FeatureData repository. Kept out of git
(see .gitignore) because it is ~90 MB; download once, locally or in CI."""
import os
import urllib.request

DEST = os.environ.get("UGR16_DIR", "data/ugr16")
FILES = ["UGR16v1.Xtrain.csv", "UGR16v1.Xtest.csv",
         "UGR16v1.Ytrain.csv", "UGR16v1.Ytest.csv"]
BASES = [
    "https://raw.githubusercontent.com/josecamachop/UGR16_FeatureData/master/csv/",
    "https://raw.githubusercontent.com/josecamachop/UGR16_FeatureData/main/csv/",
]


def fetch(name):
    target = os.path.join(DEST, name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"  have {name}")
        return
    for base in BASES:
        try:
            urllib.request.urlretrieve(base + name, target)
            print(f"  downloaded {name} ({os.path.getsize(target)//1024} KB)")
            return
        except Exception:
            continue
    raise RuntimeError(f"could not download {name} from any known branch")


def main():
    os.makedirs(DEST, exist_ok=True)
    print(f"fetching UGR'16 v1 into {DEST}/")
    for f in FILES:
        fetch(f)
    print("done")


if __name__ == "__main__":
    main()
