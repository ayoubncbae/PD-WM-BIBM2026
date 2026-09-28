import torch
from torch.utils.data import Dataset
from .image_loading import load_bag


SLOT_NAMES=["none","low","medium","high"]


def slot_id(value):
    value=str(value).lower()
    return SLOT_NAMES.index(value) if value in SLOT_NAMES else -1


class PatientDataset(Dataset):
    def __init__(self,frame,image_manifest,text_encoder,candidate_vectors,cfg):
        self.df=frame.reset_index(drop=True); self.images=image_manifest; self.enc=text_encoder
        self.candidates=candidate_vectors; self.cfg=cfg
    def __len__(self): return len(self.df)
    def __getitem__(self,index):
        row=self.df.iloc[index]; sub=self.images[(self.images.subject_id==row.subject_id)&(self.images.readable)]
        mri_paths=sub[sub.modality=="MRI"].path.tolist(); dat_paths=sub[sub.modality=="DaT"].path.tolist()
        mri,mri_mask=load_bag(mri_paths[:1],self.cfg.BAG_SIZE,self.cfg.IMAGE_SIZE)
        dat,dat_mask=load_bag(dat_paths[:1],self.cfg.BAG_SIZE,self.cfg.IMAGE_SIZE)
        descriptors=[
            row.narrative,"levodopa",f"Clinical context for considering levodopa for: {row.narrative}",
            "Potential motor symptom management context for levodopa without predicting an outcome.",
            f"Applicability context derived only from target-safe profile: {row.narrative}"
        ]
        vectors=self.enc.encode(descriptors)
        return {"subject_id":row.subject_id,"mri":mri,"mri_mask":mri_mask,"dat":dat,"dat_mask":dat_mask,
                "narrative":vectors[0],"medicine":vectors[1],"pre":vectors[2],"effect":vectors[3],"applicability":vectors[4],
                "candidates":self.candidates,"severity":torch.tensor(int(row.severity)),
                "management":torch.tensor(int(row.management_id)),"use":torch.tensor(int(row.use_label)),
                "dose":torch.tensor(slot_id(row.dose_label)),"intensity":torch.tensor(slot_id(row.intensity_label))}
