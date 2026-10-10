import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from src.components.optuna_model_selection.model import MRIFlexAttentionUNet, CorrectionModel
import cv2
    
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from ipywidgets import interact, IntSlider, Dropdown


from torch.utils.data import Dataset, DataLoader
class BaseModelDataset(Dataset):
    def __init__(self, mri_dir: Path, transform = None):
        
        files = sorted(file for file in mri_dir.iterdir())
        self.ds = [files[0], *files, files[-1]]
        self.transform = transform

    
    def __len__(self):

        return len(self.ds)-2

            

    def __getitem__(self, index): 
        
        img_0 = cv2.imread(self.ds[index], cv2.IMREAD_UNCHANGED)
        img_1 = cv2.imread(self.ds[index+1], cv2.IMREAD_UNCHANGED)
        img_2 = cv2.imread(self.ds[index+2], cv2.IMREAD_UNCHANGED)
        
        img = np.stack([img_0, img_1, img_2])
        min_img = img.min()
        max_img = img.max()
        img = (img - min_img)/(max_img - min_img + 1e-8)
        img = torch.from_numpy(img).to(torch.float32)

        
        if self.transform:
            img = self.transform(img)
           
        

        return img, index



class MRISegmentationModel:
    def __init__(self, config):


        self.device = config.device

        state_dict = torch.load(config.base_model_path, 
                                'cpu', weights_only=False)['state_dict']
        inner_state_dict = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
        
        hparams = torch.load(config.base_model_path, 'cpu', weights_only= False)['hyper_parameters']
        base_model = MRIFlexAttentionUNet(
            inp_channels=hparams['inp_channels'],
            first_conv_out_channels = hparams['first_conv_out_channels'],
            num_classes=3,
            depth = hparams['depth'],
            n_encoder_conv_layers = hparams['n_encoder_conv_layers'],
            n_decoder_conv_layers = hparams['n_decoder_conv_layers'],
            kernel_sizes = hparams['kernel_sizes']
        )
        base_model.load_state_dict(inner_state_dict)


        state_dict = torch.load(config.correction_model_path, 
                                'cpu', weights_only=False)['state_dict']
        inner_state_dict = {
            k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
        }
        
        hparams = torch.load(config.correction_model_path, 'cpu', weights_only= False)['hyper_parameters']
        correction_model = CorrectionModel(
            inp_channels=hparams['inp_channels'],
            first_conv_out_channels = hparams['first_conv_out_channels'],
            num_classes=3,
            depth = hparams['depth'],
            n_encoder_conv_layers = hparams['n_encoder_conv_layers'],
            n_decoder_conv_layers = hparams['n_decoder_conv_layers'],
            kernel_sizes = hparams['kernel_sizes']
        )

        correction_model.load_state_dict(inner_state_dict)


        self.base_model = base_model.to(self.device).eval()
        self.correction_model = correction_model.to(self.device).eval()

        if config.base_thresholds_path is not None:
            base_thresholds = np.load(config.base_thresholds_path)

        else:
            base_thresholds = np.array([0.5,0.5,0.5])

        if config.correction_thresholds_path is not None:
            correction_thresholds = np.load(config.correction_thresholds_path)
        else:
            correction_thresholds = np.array([0.97866, 0.98422, 0.98606])

        self.base_thresholds = torch.as_tensor(base_thresholds, dtype=torch.float32, device=config.device).view(1, -1, 1, 1)
        self.correction_thresholds = torch.as_tensor(correction_thresholds, dtype=torch.float32, device=config.device).view(1, -1, 1, 1)


        self.config = config
        
        
    

    @torch.no_grad()
    def __call__(self, mri_dir):


        base_dataset = BaseModelDataset(mri_dir=mri_dir, transform=self.config.dev_transform)
        inp, _ = next(iter(base_dataset))
        self.out_shape = (3, inp.shape[1], inp.shape[2])
        self.len = len(base_dataset)
        self.base_dataloader = DataLoader(base_dataset, self.config.batch_size, False, num_workers=0)

        correction_imgs = []
        correction_inp_masks = []
        
        summed_masks = torch.zeros(self.out_shape, dtype = int, device = self.device)

        for inp, idx in self.base_dataloader:
            inp = inp.to(self.device)
            
            correction_imgs.append(inp[:,1].cpu())

            out = self.base_model(inp)
            probs = torch.sigmoid(out) 
            out = out.cpu().to(torch.float16)
            correction_inp_masks.append(out.to(torch.float16))

            summed_masks += (probs >= self.base_thresholds).int().sum(dim = 0)

    

        summed_masks = summed_masks.float() / torch.Tensor([74,61,45], device = self.device).view(3,1,1).float()
        self.summed_masks = summed_masks.unsqueeze(0).expand(self.config.batch_size, *self.out_shape)

        final_logits = []
        for imgs, masks in zip(correction_imgs, correction_inp_masks):
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            print(imgs.shape, masks.shape, self.summed_masks.shape)
            logits = self.correction_model((imgs, masks, self.summed_masks))
            final_logits.append(logits)

        self.final_logits =torch.cat(final_logits, dim = 0)
        final_probs = torch.sigmoid(self.final_logits)
        self.final_masks = (final_probs >= self.correction_thresholds).bool().cpu()

        self.slices = torch.cat(correction_imgs, dim = 0)
        return self.final_masks, (torch.cat(correction_inp_masks, dim = 0) >= self.base_thresholds).bool()









    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)


    def visualize_mask_3d(self,
        masks = None,
        class_names=("Large bowel", "Small bowel", "Stomach"),
        sigma=1.0,
        threshold=0.5,
        step_size=1,
        spacing=(1.0, 1.0, 1.0),
    ):
        """
        Display each class in its own interactive 3D figure.

        masks: (N, C, H, W) or (C, N, H, W)
        """

        if masks == None:
            masks = self.final_masks

        masks = self._to_numpy(masks)

        if masks.ndim != 4:
            raise ValueError(f"Expected 4D masks, got {masks.shape}")

        # Model output is normally (N, C, H, W).
        # Convert to (C, N, H, W).
        if masks.shape[1] <= 10:
            masks = np.moveaxis(masks, 1, 0)

        colors = ["red", "limegreen", "dodgerblue",
                "orange", "magenta"]

        figures = []

        for c, volume in enumerate(masks):
            volume = volume.astype(np.float32)

            if not np.isfinite(volume).all():
                raise ValueError(f"Class {c} contains NaN or infinity.")

            # Smooth a copy only; don't alter original predictions.
            smoothed = (
                gaussian_filter(volume, sigma=sigma)
                if sigma > 0 else volume
            )

            binary = smoothed >= threshold

            name = (
                class_names[c]
                if c < len(class_names)
                else f"Class {c}"
            )

            if not binary.any():
                print(f"{name}: empty mask, skipped.")
                continue

            if binary.all():
                print(f"{name}: entire volume is foreground, skipped.")
                continue

            vertices, faces, _, _ = marching_cubes(
                smoothed,
                level=threshold,
                spacing=spacing,
                step_size=step_size,
                allow_degenerate=False,
            )

            # vertices columns are slice, height, width
            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=vertices[:, 2],
                        y=vertices[:, 1],
                        z=vertices[:, 0],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color=colors[c % len(colors)],
                        opacity=0.85,
                        name=name,
                        flatshading=False,
                        hovertemplate=(
                            "Width: %{x:.1f}<br>"
                            "Height: %{y:.1f}<br>"
                            "Slice: %{z:.1f}<extra></extra>"
                        ),
                    )
                ]
            )

            fig.update_layout(
                title=f"3D Segmentation — {name}",
                scene=dict(
                    xaxis_title="Width",
                    yaxis_title="Height",
                    zaxis_title="Slice",
                    aspectmode="data",
                ),
                margin=dict(l=0, r=0, t=50, b=0),
            )

            figures.append(fig)
            fig.show()

        return figures


    def visualize_mri_overlay(self,
        mri = None,
        masks = None,
        class_names=("Large bowel", "Small bowel", "Stomach"),
        colors=("red", "lime", "dodgerblue"),
        alpha=0.4,
        threshold=0.5,
        sigma=0,
    ):
        """
        Interactive slice viewer with the original MRI and a
        colored overlay for one selected segmentation class.

        mri:   (N, H, W), grayscale MRI volume
        masks: (N, C, H, W) or (C, N, H, W)

        sigma=0 shows the original mask without smoothing.
        """
        if mri == None:
            mri = self.slices

        if masks == None:
            masks = self.final_masks

        
        mri = self._to_numpy(mri)
        masks = self._to_numpy(masks)

        if mri.ndim != 3:
            raise ValueError(f"Expected MRI (N,H,W), got {mri.shape}")

        if masks.ndim != 4:
            raise ValueError(f"Expected 4D masks, got {masks.shape}")

        # Convert model output (N, C, H, W) to (C, N, H, W).
        if masks.shape[0] == mri.shape[0]:
            masks = np.moveaxis(masks, 1, 0)

        if masks.shape[1:] != mri.shape:
            raise ValueError(
                f"MRI shape {mri.shape} doesn't match masks "
                f"volume shape {masks.shape[1:]}"
            )

        n_classes = masks.shape[0]

        def show_slice(slice_idx=0, class_name=None):
            if class_name is None:
                class_idx = 0
            else:
                class_idx = class_names.index(class_name)

            img = mri[slice_idx].astype(np.float32)
            mask_volume = masks[class_idx].astype(np.float32)

            # Normalize only for display.
            lo, hi = np.percentile(img, [1, 99])
            img_display = np.clip(
                (img - lo) / (hi - lo + 1e-8), 0, 1
            )

            if sigma > 0:
                mask_volume = gaussian_filter(
                    mask_volume, sigma=sigma
                )

            mask_slice = mask_volume[slice_idx] >= threshold

            fig = go.Figure()

            # Original MRI in grayscale.
            fig.add_trace(
                go.Heatmap(
                    z=img_display,
                    colorscale="Gray",
                    zmin=0,
                    zmax=1,
                    showscale=False,
                    name="Original MRI",
                    hovertemplate=(
                        "x: %{x}<br>y: %{y}<br>"
                        "Intensity: %{z:.3f}<extra></extra>"
                    ),
                )
            )

            # Transparent overlay: NaNs are transparent.
            overlay = np.where(mask_slice, 1.0, np.nan)

            fig.add_trace(
                go.Heatmap(
                    z=overlay,
                    colorscale=[
                        [0, colors[class_idx % len(colors)]],
                        [1, colors[class_idx % len(colors)]],
                    ],
                    opacity=alpha,
                    showscale=False,
                    name=class_names[class_idx],
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                title=(
                    f"MRI slice {slice_idx} — "
                    f"{class_names[class_idx]}"
                ),
                xaxis_title="Width",
                yaxis_title="Height",
                yaxis=dict(autorange="reversed", scaleanchor="x"),
                margin=dict(l=20, r=20, t=50, b=20),
            )

            fig.show()

        # Use an interactive notebook widget to select slice and class.
        interact(
            show_slice,
            slice_idx=IntSlider(
                min=0, max=mri.shape[0] - 1, value=0,
                description="Slice",
            ),
            class_name=Dropdown(
                options=list(class_names[:n_classes]),
                value=class_names[0],
                description="Class",
            ),
        )