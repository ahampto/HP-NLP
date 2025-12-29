import json
import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize character names just in case
    if "Character" in df.columns:
        df["Character"] = df["Character"].astype(str).str.strip()
    return df

def load_descriptions(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_descriptions(df: pd.DataFrame, desc_map: dict) -> pd.DataFrame:
    # Desc map format: { Character: {sloan, tagline, points} }
    def get_points(name: str):
        item = desc_map.get(name, {})
        pts = item.get("points", None)
        if isinstance(pts, list) and len(pts) > 0:
            return pts
        return None

    def get_tagline(name: str):
        item = desc_map.get(name, {})
        tl = item.get("tagline", None)
        if isinstance(tl, str) and tl.strip():
            return tl.strip()
        return None

    df["tagline_override"] = df["Character"].map(get_tagline)
    df["points_override"] = df["Character"].map(get_points)
    return df

def default_tagline(sloan: str, cluster: str = "") -> str:
    s = (sloan or "").strip()
    if not s:
        return cluster or "Character profile"
    first = s[0].lower()
    style = "Reactive" if first == "r" else "Agentic" if first == "s" else "Profile"
    affect = "emotion-driven" if "L" in s else "steady"
    structure = "structured" if "O" in s else ("flexible" if "U" in s else "")
    bits = [style, affect]
    if structure:
        bits.append(structure)
    return ", ".join([b for b in bits if b][:3])

def validate_names(df: pd.DataFrame, desc_map: dict) -> dict:
    df_names = set(df["Character"].dropna().astype(str).str.strip())
    json_names = set(k.strip() for k in desc_map.keys())
    return {
        "in_json_not_in_df": sorted(list(json_names - df_names)),
        "in_df_not_in_json": sorted(list(df_names - json_names))
    }
