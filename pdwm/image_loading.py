from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from PIL import Image


def _normalize(arr):
    arr=np.nan_to_num(arr.astype("float32")); lo,hi=np.percentile(arr,[1,99])
    if hi<=lo: return np.zeros_like(arr,dtype="float32")
    return np.clip((arr-lo)/(hi-lo),0,1).astype("float32")


def load_instances(path, count, size):
    low=str(path).lower()
    if low.endswith((".nii", ".nii.gz")):
        def selected_slices(image):
            shape=image.shape[:3]; axis=int(np.argmax(shape)); n=shape[axis]
            indices=np.linspace(max(0,int(n*.3)),min(n-1,int(n*.7)),count).astype(int)
            result=[]
            for i in indices:
                key=[slice(None)]*len(image.shape); key[axis]=int(i)
                arr=np.asarray(image.dataobj[tuple(key)])
                if arr.ndim>2: arr=arr.mean(axis=tuple(range(2,arr.ndim)))
                result.append(arr)
            return result
        try:
            slices=selected_slices(nib.load(str(path)))
        except nib.filebasedimages.ImageFileError:
            # Some supplied files are valid uncompressed NIfTI payloads with a .nii.gz suffix.
            # Read the header/payload directly without rewriting the user's raw data.
            with open(path,"rb") as handle:
                holder=nib.FileHolder(fileobj=handle)
                image=nib.Nifti1Image.from_file_map({"header":holder,"image":holder})
                slices=selected_slices(image)
    else:
        arr=np.asarray(Image.open(path).convert("L")); slices=[arr]*count
    tensors=[]
    for arr in slices:
        t=torch.from_numpy(_normalize(arr).copy())[None,None]
        t=F.interpolate(t,(size,size),mode="bilinear",align_corners=False)[0]
        tensors.append(t.repeat(3,1,1))
    return torch.stack(tensors)


def load_bag(paths, count, size):
    if not paths: return torch.zeros(count,3,size,size), torch.zeros(count)
    per=max(1,count//len(paths)); pieces=[]
    for p in paths:
        try: pieces.extend(load_instances(p,per,size))
        except (OSError,ValueError,nib.filebasedimages.ImageFileError): continue
    if not pieces: return torch.zeros(count,3,size,size), torch.zeros(count)
    pieces=pieces[:count]
    while len(pieces)<count: pieces.append(pieces[-1].clone())
    return torch.stack(pieces), torch.ones(count)
