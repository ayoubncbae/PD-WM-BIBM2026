import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from .data import prepare_records,split_subjects,SAFE_FIELDS
from .encoders import MedCPTTextEncoder
from .dataset import PatientDataset,SLOT_NAMES
from .model import PDWorldModel
from .losses import joint_loss
from .metrics import severity_metrics,classification_slot
from .utils import seed_everything,write_json,environment_text


def _move(batch,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def _epoch(model,loader,cfg,optimizer=None):
    model.train(optimizer is not None); totals=[]; terms=[]
    for batch in loader:
        batch=_move(batch,cfg.DEVICE)
        if optimizer: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            out=model(batch); loss,pieces=joint_loss(out,batch,cfg)
            if optimizer: loss.backward(); optimizer.step()
        totals.append(loss.item()); terms.append(pieces)
    mean={k:float(np.mean([x[k] for x in terms])) for k in terms[0]} if terms else {"total":float("nan")}
    return float(np.mean(totals)) if totals else float("nan"),mean


def _predict(model,loader,cfg,candidates):
    rows=[]; routers=[]; model.eval()
    with torch.no_grad():
        for batch in loader:
            device_batch=_move(batch,cfg.DEVICE); out=model(device_batch)
            sp=out["severity_probs"].cpu().numpy(); mp=out["management_probs"].cpu().numpy()
            use=out["use_logits"].argmax(-1).cpu().numpy(); dose=out["dose_logits"].argmax(-1).cpu().numpy(); intensity=out["intensity_logits"].argmax(-1).cpu().numpy()
            for i,sid in enumerate(batch["subject_id"]):
                true_m=int(batch["management"][i]); pred_m=int(mp[i].argmax())
                rows.append({"subject_id":sid,"true_severity":int(batch["severity"][i]),"pred_severity":int(sp[i].argmax()),
                    "severity_prob_0":sp[i,0],"severity_prob_1":sp[i,1],"severity_prob_2":sp[i,2],"medicine":"levodopa",
                    "true_use":int(batch["use"][i]),"pred_use":int(use[i]),"true_dose":int(batch["dose"][i]),"pred_dose":int(dose[i]),
                    "true_intensity":int(batch["intensity"][i]),"pred_intensity":int(intensity[i]),
                    "true_management_state":candidates[true_m] if true_m>=0 else "unseen_in_training_candidates",
                    "pred_management_state":candidates[pred_m],"management_top1_probability":float(mp[i,pred_m]),
                    "management_true_id":true_m,"management_pred_id":pred_m,"management_top3_ids":np.argsort(-mp[i])[:3].tolist()})
                routers.append({"subject_id":sid,**{f"expert_{j}":float(x) for j,x in enumerate(out["router"][i].cpu())}})
    return pd.DataFrame(rows),pd.DataFrame(routers)


def _plots(history,pred,routers,figdir):
    figdir.mkdir(parents=True,exist_ok=True)
    plt.figure(); plt.plot(history.epoch,history.train_loss,label="train"); plt.plot(history.epoch,history.val_loss,label="validation"); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout(); plt.savefig(figdir/"training_loss.png"); plt.close()
    cm=confusion_matrix(pred.true_severity,pred.pred_severity,labels=[0,1,2]); plt.figure(); plt.imshow(cm,cmap="Blues"); plt.colorbar(); plt.xlabel("predicted"); plt.ylabel("true"); plt.xticks(range(3)); plt.yticks(range(3)); plt.tight_layout(); plt.savefig(figdir/"severity_confusion_matrix.png"); plt.close()
    plt.figure(); probs=pred[["severity_prob_0","severity_prob_1","severity_prob_2"]].to_numpy(); plt.imshow(probs,aspect="auto",vmin=0,vmax=1,cmap="viridis"); plt.colorbar(); plt.yticks(range(len(pred)),pred.subject_id); plt.xlabel("severity class"); plt.tight_layout(); plt.savefig(figdir/"severity_probabilities.png"); plt.close()
    valid=pred.management_true_id>=0
    if valid.any():
        labels=sorted(set(pred.loc[valid,"management_true_id"])|set(pred.loc[valid,"management_pred_id"])); mcm=confusion_matrix(pred.loc[valid,"management_true_id"],pred.loc[valid,"management_pred_id"],labels=labels)
        plt.figure(); plt.imshow(mcm,cmap="Purples"); plt.colorbar(); plt.xlabel("predicted state id"); plt.ylabel("true state id"); plt.tight_layout(); plt.savefig(figdir/"management_confusion_matrix.png"); plt.close()
    plt.figure(); x=np.arange(len(pred)); plt.plot(x,pred.true_severity,"o",label="true"); plt.plot(x,pred.pred_severity,"x",label="predicted"); plt.xticks(x,pred.subject_id); plt.xlabel("subject"); plt.ylabel("severity"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"subject_prediction_summary.png"); plt.close()
    if not routers.empty:
        routers.set_index("subject_id").plot(kind="bar",ylim=(0,1)); plt.ylabel("router weight"); plt.tight_layout(); plt.savefig(figdir/"router_weights.png"); plt.close()


def run(cfg,llm):
    seed_everything(cfg.SEED); cfg.OUTPUT_DIR.mkdir(parents=True,exist_ok=True); (cfg.OUTPUT_DIR/"figures").mkdir(exist_ok=True)
    df,images,raw,all_images=prepare_records(cfg,llm)
    management_narratives=[]
    for _,row in df.iterrows():
        fallback=(f"Ground-truth management state for output supervision: medicine={row.medicine}, use="
                  f"{row.management_state.split('|')[1]}, dose={row.dose_label}, intensity={row.intensity_label}.")
        prompt=("Using the patient illness narrative and ground-truth fields below, write a concise management state description. "
                "Do not create prescriptions, future outcomes, or unsupported recommendations. This text is output supervision only.\n"
                f"Narrative: {row.narrative}\nStructured fields: {fallback}")
        management_narratives.append(llm.generate("management_supervision",row.subject_id,prompt,fallback))
    df["management_narrative"]=management_narratives; df["split"]=split_subjects(df,cfg.SEED)
    train=df[df.split=="train"]; candidates=sorted(train.management_state.unique().tolist()); cmap={x:i for i,x in enumerate(candidates)}
    df["management_id"]=df.management_state.map(cmap).fillna(-1).astype(int)
    text_encoder=MedCPTTextEncoder(cfg.TEXT_DIM,cfg.MEDCPT_BACKEND,cfg.MEDCPT_MODEL_PATH)
    candidate_vectors=text_encoder.encode(candidates)
    loaders={}
    for split in ["train","val","test"]:
        ds=PatientDataset(df[df.split==split],images,text_encoder,candidate_vectors,cfg)
        loaders[split]=DataLoader(ds,batch_size=cfg.BATCH_SIZE,shuffle=(split=="train"),num_workers=0)

    actual_mri=int((images.modality=="MRI").sum()); actual_dat=int((images.modality=="DaT").sum())
    audit={"csv_rows":len(raw),"csv_columns":len(raw.columns),"bids_subjects_with_supported_images":len(set(all_images.subject_id)),
        "matched_subjects_available":len(set(raw.subject_id)&set(all_images.subject_id)),"subjects_selected":len(df),
        "selected_subject_ids":df.subject_id.tolist(),"severity_counts":df.severity.value_counts().sort_index().to_dict(),
        "actual_mri_files_selected":actual_mri,"actual_dat_files_selected":actual_dat,"other_imaging_files_selected":int((images.modality=="other").sum()),
        "readable_mri_files_selected":int(((images.modality=="MRI")&images.readable).sum()),
        "unreadable_or_truncated_mri_files_selected":int(((images.modality=="MRI")&~images.readable).sum()),
        "nifti_container_notice":"Supplied .nii.gz payloads are uncompressed NIfTI; loader detects this extension/container mismatch and reads them without modifying raw files.",
        "dat_mismatch_notice":"No DaT/SPECT-labelled files exist under the supplied OpenNeuro root; CSV dat_image_count appears to describe another collection. DaT branch used explicit missing-modality bags.",
        "clinical_columns_used_for_target_safe_text":[c for c in SAFE_FIELDS if c in raw.columns],
        "severity_source_columns":["disease_label","Stage (I-V)","Stage (I-V).1","UPDRS III (0-126)"],
        "management_source_columns":["target_ldopa_use","target_ldopa_dose_category","target_medication_intensity"]}
    write_json(cfg.OUTPUT_DIR/"dataset_audit.json",audit)
    print("DATASET AUDIT:",json.dumps(audit,indent=2))
    print("Train IDs:",df[df.split=="train"].subject_id.tolist()); print("Validation IDs:",df[df.split=="val"].subject_id.tolist()); print("Test IDs:",df[df.split=="test"].subject_id.tolist())

    model=PDWorldModel(cfg).to(cfg.DEVICE); optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.LR)
    history=[]; best=float("inf"); best_path=cfg.OUTPUT_DIR/"best_model.pt"
    for epoch in range(1,cfg.EPOCHS+1):
        train_loss,terms=_epoch(model,loaders["train"],cfg,optimizer); val_loss,_=_epoch(model,loaders["val"],cfg)
        history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,**{f"train_{k}":v for k,v in terms.items()}})
        print(f"Epoch {epoch}/{cfg.EPOCHS} train={train_loss:.4f} val={val_loss:.4f} terms={terms}")
        if val_loss<best: best=val_loss; torch.save(model.state_dict(),best_path)
    model.load_state_dict(torch.load(best_path,map_location=cfg.DEVICE,weights_only=True))
    pred,routers=_predict(model,loaders["test"],cfg,candidates)
    pred["true_management_state"]=pred.subject_id.map(dict(zip(df.subject_id,df.management_state)))
    sev=severity_metrics(pred.true_severity,pred.pred_severity,pred[["severity_prob_0","severity_prob_1","severity_prob_2"]])
    valid=pred.management_true_id>=0
    mgmt={"candidate_state_count":len(candidates),"test_subject_count":len(pred),"test_states_covered_by_training_candidates":int(valid.sum()),"test_states_unseen_in_training_candidates":int((~valid).sum()),
          "structured_state_top1_accuracy":float(accuracy_score(pred.loc[valid,"management_true_id"],pred.loc[valid,"management_pred_id"])) if valid.any() else float("nan")}
    if len(candidates)>=3 and valid.any(): mgmt["structured_state_top3_accuracy"]=float(np.mean([t in ids for t,ids in zip(pred.loc[valid,"management_true_id"],pred.loc[valid,"management_top3_ids"])]))
    else: mgmt["structured_state_top3_accuracy"]=float("nan")
    for col,pcol,name in [("true_use","pred_use","medicine_use"),("true_dose","pred_dose","dose"),("true_intensity","pred_intensity","intensity")]: mgmt.update(classification_slot(pred[col],pred[pcol],name))

    verbal=[]
    for i,row in pred.head(3).iterrows():
        fallback=(f"Predicted severity class {row.pred_severity}. For the predefined medicine levodopa, the structured model predicts "
                  f"use={row.pred_use}, dose={SLOT_NAMES[row.pred_dose]}, and management intensity={SLOT_NAMES[row.pred_intensity]}. "
                  "This is a model output for research verification, not a clinical recommendation.")
        prompt=("Verbalize only this structured model prediction. Do not prescribe, change the medicine or dose, invent details, recommend care, or predict outcomes.\n"+fallback)
        text=llm.generate("verbalization",row.subject_id,prompt,fallback); pred.loc[i,"generated_management_text"]=text; verbal.append({"subject_id":row.subject_id,"text":text})

    df[["subject_id","disease_label","severity","medicine","use_label","dose_label","intensity_label","management_state","management_narrative","split"]].to_csv(cfg.OUTPUT_DIR/"subject_manifest.csv",index=False)
    images.to_csv(cfg.OUTPUT_DIR/"image_manifest.csv",index=False); df[["subject_id","severity","split"]].to_csv(cfg.OUTPUT_DIR/"split_manifest.csv",index=False)
    hist=pd.DataFrame(history); hist.to_csv(cfg.OUTPUT_DIR/"training_history.csv",index=False); pred.drop(columns=["management_true_id","management_pred_id","management_top3_ids"]).to_csv(cfg.OUTPUT_DIR/"predictions.csv",index=False)
    routers.to_csv(cfg.OUTPUT_DIR/"router_weights.csv",index=False); pd.DataFrame({"management_id":range(len(candidates)),"state":candidates}).to_csv(cfg.OUTPUT_DIR/"candidate_management_states.csv",index=False)
    write_json(cfg.OUTPUT_DIR/"severity_metrics.json",sev); write_json(cfg.OUTPUT_DIR/"management_metrics.json",mgmt); write_json(cfg.OUTPUT_DIR/"example_verbalizations.json",verbal); write_json(cfg.OUTPUT_DIR/"run_config.json",cfg.serializable())
    (cfg.OUTPUT_DIR/"environment_info.txt").write_text(environment_text(),encoding="utf-8")
    action_cache={r.subject_id:{"precondition":f"Target-safe clinical context for predefined levodopa: {r.narrative}","effect":"Motor management context; no outcome asserted.","applicability":f"Applicability context from target-safe profile: {r.narrative}"} for _,r in df.iterrows()}
    write_json(cfg.CACHE_DIR/"medicine_action_descriptors.json",action_cache)
    _plots(hist,pred,routers,cfg.OUTPUT_DIR/"figures")
    return {"subjects":len(df),"splits":df.split.value_counts().to_dict(),"audit":audit,"severity":sev,"management":mgmt,"llm":llm.source_counts}
