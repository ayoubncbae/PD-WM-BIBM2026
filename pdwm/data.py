import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from .utils import clean


SAFE_FIELDS = [
    "Sex (M/F)", "Age (=last seen-D)", "Symptoms started", "Disease Duration (years) (today-E)",
    "Lateral (body)", "UPDRS I Nm-EDL (0-42)", "UPDRS II M-EDL (0-42)", "UPDRS IV (0-24)",
    "Timed -Upper R. (number of repeats)", "Timed -Upper L. (number of repeats)", "Timed-walk (sec)",
    "# of steps", "FOG (0-24)", "MMSE (0-30)", "Halluc", "Hyperkinesias", "smell", "RBD",
    "Const", "family", "ET", "Smoking 0=N 1=Y 2=cess", "Diabetes", "Depres"
]


def norm_sid(value):
    try: return str(int(float(value)))
    except Exception: return clean(value)


def severity(row):
    if clean(row.get("disease_label")).upper() != "PD": return 0
    stage = pd.to_numeric(row.get("Stage (I-V)"), errors="coerce")
    if pd.isna(stage): stage = pd.to_numeric(row.get("Stage (I-V).1"), errors="coerce")
    if pd.notna(stage): return 1 if stage < 3 else 2
    updrs = pd.to_numeric(row.get("UPDRS III (0-126)"), errors="coerce")
    return 1 if pd.isna(updrs) or updrs <= 32 else 2


def narrative_fallback(row):
    labels = {
        "Sex (M/F)":"sex", "Age (=last seen-D)":"age", "Symptoms started":"symptom history",
        "Disease Duration (years) (today-E)":"recorded illness duration", "Lateral (body)":"body side",
        "UPDRS I Nm-EDL (0-42)":"non-motor daily-living score", "UPDRS II M-EDL (0-42)":"motor daily-living score",
        "UPDRS IV (0-24)":"motor complication score", "Timed-walk (sec)":"timed walk seconds",
        "FOG (0-24)":"freezing-of-gait score", "MMSE (0-30)":"cognitive score", "Halluc":"hallucination indicator",
        "smell":"smell status", "RBD":"RBD", "Const":"constipation", "family":"family history",
        "ET":"essential tremor", "Smoking 0=N 1=Y 2=cess":"smoking", "Diabetes":"diabetes", "Depres":"depression"
    }
    parts = [f"{labels.get(c,c)} {clean(row.get(c))}" for c in SAFE_FIELDS if clean(row.get(c)) != "unknown"]
    return "Clinical profile: " + "; ".join(parts) + "."


def discover_images(root, valid_subjects):
    records = []
    for path in Path(root).rglob("*"):
        if not path.is_file() or not (path.name.lower().endswith((".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".dcm"))): continue
        match = re.search(r"sub-0*(\d+)", str(path), re.I)
        if not match or match.group(1) not in valid_subjects: continue
        low = path.name.lower()
        modality = "MRI" if ("t1w" in low or "anat" in str(path).lower()) else ("DaT" if "dat" in low or "spect" in low else "other")
        readable=True
        if ".nii" in low:
            try:
                try: image=nib.load(str(path))
                except nib.filebasedimages.ImageFileError:
                    handle=open(path,"rb"); holder=nib.FileHolder(fileobj=handle); image=nib.Nifti1Image.from_file_map({"header":holder,"image":holder})
                key=tuple(int(x//2) for x in image.shape)
                _=image.dataobj[key]
                if 'handle' in locals(): handle.close(); del handle
            except Exception:
                readable=False
                if 'handle' in locals(): handle.close(); del handle
        records.append({"subject_id":match.group(1), "modality":modality, "path":str(path.resolve()), "format":"nifti" if ".nii" in low else path.suffix.lower(),"readable":readable})
    return pd.DataFrame(records, columns=["subject_id","modality","path","format","readable"])


def prepare_records(cfg, llm):
    raw = pd.read_csv(cfg.PATIENT_CSV); raw["subject_id"] = raw["subject_id"].map(norm_sid)
    image_df = discover_images(cfg.DATA_ROOT, set(raw.subject_id))
    matched = sorted(set(raw.subject_id) & set(image_df.subject_id), key=lambda x:int(x))
    df = raw[raw.subject_id.isin(matched)].copy(); df["severity"] = df.apply(severity, axis=1)
    # Deterministic diversity-first sample: round-robin by severity, then fill to MAX_SUBJECTS.
    chosen=[]; selection_limit=min(cfg.MAX_SUBJECTS,len(df)) if cfg.QUICK_RUN else len(df)
    groups={k:list(g.sort_values("subject_id", key=lambda s:s.astype(int)).index) for k,g in df.groupby("severity")}
    while len(chosen) < selection_limit and any(groups.values()):
        for k in [0,1,2]:
            if groups.get(k) and len(chosen)<selection_limit: chosen.append(groups[k].pop(0))
    df=df.loc[chosen].reset_index(drop=True)
    narratives=[]
    for _,row in df.iterrows():
        fallback=narrative_fallback(row)
        prompt=("Write a concise illness narrative using only these target-safe variables. Never mention diagnosis, stage, "
                "UPDRS-III, medication use, L-dopa use, dose, intensity, or management targets.\n"+fallback)
        narratives.append(llm.generate("patient", row.subject_id, prompt, fallback))
    df["narrative"]=narratives
    df["medicine"]="levodopa"
    use=pd.to_numeric(df.get("target_ldopa_use"),errors="coerce")
    df["use_label"]=use.fillna(-1).astype(int)
    df["dose_label"]=df.get("target_ldopa_dose_category",pd.Series("unknown",index=df.index)).map(lambda x:clean(x).lower())
    df["intensity_label"]=df.get("target_medication_intensity",pd.Series("unknown",index=df.index)).map(lambda x:clean(x).lower())
    df["management_state"]=[f"levodopa|{('used' if u==1 else 'not_used' if u==0 else 'unknown')}|{d}|{i}" for u,d,i in zip(df.use_label,df.dose_label,df.intensity_label)]
    selected_images=image_df[image_df.subject_id.isin(df.subject_id)].copy()
    return df, selected_images, raw, image_df


def split_subjects(df, seed=42):
    idx=np.arange(len(df)); labels=df.severity.to_numpy()
    try: train, temp=train_test_split(idx,test_size=0.4,random_state=seed,stratify=labels)
    except ValueError: train,temp=train_test_split(idx,test_size=0.4,random_state=seed)
    try: val,test=train_test_split(temp,test_size=0.5,random_state=seed,stratify=labels[temp])
    except ValueError: val,test=train_test_split(temp,test_size=0.5,random_state=seed)
    out=np.full(len(df),"train",dtype=object); out[val]="val"; out[test]="test"
    return out
