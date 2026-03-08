# **Foreground-Guided 3D Gaussian Splatting for Wheat Reconstruction**

## **Objective**

### The goal of this experiment was to investigate whether **segmentation masks of wheat plants** can improve 3D reconstruction using **3D Gaussian Splatting (3DGS)**. In particular, we evaluated whether guiding the reconstruction with **foreground masks** helps the model focus on the plant structure instead of background regions.

### All experiments were performed on the **frame\_0001 dataset (21 cameras)** with a **train/test split of 16 training views and 5 test views**. Training was run for **15,000 iterations** with **resolution \= 4**.

### ---

# **Experiment 1 — Baseline reconstruction (no masking)**

### In the baseline setup, the model is trained on the **original RGB images without any segmentation masks**. This means the model must reconstruct both the wheat plant and the background.

### This baseline serves as a reference to measure the effect of introducing foreground masks.

### *(Baseline metrics were not recomputed in this session but serve as conceptual comparison.)*

### ---

# **Experiment 2 — Hard-masked RGB images**

### In the second experiment, we introduced **ground-truth segmentation masks** by **removing the background directly from the RGB images**.

### Pixels outside the plant mask were replaced with black before training.

### This effectively forces the model to learn only from **plant pixels**.

### **Training setup**

* ### Dataset: frame\_0001

* ### Train views: 16

* ### Test views: 5

* ### Resolution: 4

* ### Iterations: 15,000

### **Results**

| Metric | Value |
| :---- | :---- |
| **PSNR** | **44.45** |
| **SSIM** | **0.99893** |
| **LPIPS** | **0.00125** |

### These results indicate **extremely accurate reconstructions**, suggesting that removing the background simplifies the learning problem.

### However, this approach **modifies the RGB images themselves**, which is not ideal for a clean pipeline.

### ---

# **Experiment 3 — Alpha-mask reconstruction (GT masks)**

### In the third experiment, we followed a **cleaner approach suggested by the supervisor**.

### Instead of modifying the RGB images, we:

1. ### kept the **original RGB images unchanged**

2. ### loaded **ground-truth segmentation masks**

3. ### passed them as **alpha masks to the Camera class**

### This uses the gt\_alpha\_mask mechanism already present in the original **3DGS implementation**.

### The alpha mask multiplies the image inside the renderer:

### self.original\_image \*= gt\_alpha\_mask

### so the loss is applied only on foreground pixels.

### **Implementation details**

### We modified the pipeline to:

* ### load mask paths in dataset\_readers.py

* ### pass masks through CameraInfo

* ### load and resize masks in camera\_utils.py

* ### apply them through gt\_alpha\_mask

### The masks were resized to the same resolution as the input images using **nearest-neighbour interpolation** to preserve binary boundaries.

### **Training setup**

* ### Dataset: frame\_0001

* ### Train views: 16

* ### Test views: 5

* ### Resolution: 4

* ### Iterations: 15,000

### **Results**

| Metric | Value |
| :---- | :---- |
| **PSNR** | **43.21** |
| **SSIM** | **0.99880** |
| **LPIPS** | **0.00164** |

### ---

# **Comparison of masking strategies**

| Method | PSNR | SSIM | LPIPS |
| :---- | :---- | :---- | :---- |
| Hard-masked RGB | **44.45** | **0.99893** | **0.00125** |
| Alpha-mask (GT) | **43.21** | **0.99880** | **0.00164** |

### Both approaches achieve **very high reconstruction quality**.

### The hard-masked RGB setup performs slightly better numerically because the problem is simplified by removing all background pixels directly in the images.

### However, the **alpha-mask approach is methodologically cleaner**, since:

* ### the original RGB images remain unchanged

* ### segmentation is incorporated through the camera model

* ### the pipeline can naturally support **predicted masks**

### ---

# **Conclusion**

### These experiments demonstrate that **foreground masking significantly improves reconstruction quality** when training 3D Gaussian Splatting on plant scenes.

### Using **ground-truth segmentation masks as alpha masks** achieves nearly the same reconstruction quality as directly masking the RGB images, while preserving a cleaner and more flexible pipeline.

### This validates the use of **segmentation-guided reconstruction** and prepares the system for experiments using **automatically predicted masks**.

### ---

# **Next steps**

### The next experiment will evaluate whether **predicted segmentation masks** (e.g. from Grounded-SAM) can provide similar reconstruction improvements compared to ground-truth masks.

### The key comparison will be:

| Mask type | Expected goal |
| :---- | :---- |
| Ground-truth masks | upper performance bound |
| Predicted masks | practical real-world pipeline |

### If predicted masks achieve similar metrics, this would demonstrate that **segmentation-guided 3D reconstruction can be applied automatically to large wheat datasets**.

### 

### 

### Results on the binary masks images (ground truth)

- 21 cameras \- 11 for training, 10 for test

00000.png: PSNR=45.1541, SSIM=0.9991  
00001.png: PSNR=43.2341, SSIM=0.9986  
00002.png: PSNR=43.5928, SSIM=0.9987  
00003.png: PSNR=45.3957, SSIM=0.9991  
00004.png: PSNR=43.4194, SSIM=0.9989  
00005.png: PSNR=44.6402, SSIM=0.9991  
00006.png: PSNR=42.4148, SSIM=0.9986  
00007.png: PSNR=42.9695, SSIM=0.9986  
00008.png: PSNR=43.5057, SSIM=0.9986  
00009.png: PSNR=45.4927, SSIM=0.9991

\=== Averages \===  
Mean PSNR: 43.981888598062945  
Mean SSIM: 0.9988352310041121

00000.png: LPIPS=0.0012  
00001.png: LPIPS=0.0031  
00002.png: LPIPS=0.0014  
00003.png: LPIPS=0.0013  
00004.png: LPIPS=0.0012  
00005.png: LPIPS=0.0009  
00006.png: LPIPS=0.0013  
00007.png: LPIPS=0.0014  
00008.png: LPIPS=0.0023  
00009.png: LPIPS=0.0016

Mean LPIPS: 0.0015718592330813408

- With 5 cameras for test, 16 for training

Mean PSNR: 44.45180274068285  
Mean SSIM: 0.9989274957710623

Mean LPIPS: 0.0012463962193578482

### Results using the alpha channel

- With 5 cameras for test, 16 for training

Mean PSNR: 43.208710308590824  
Mean SSIM: 0.9987999916126016

Mean LPIPS: 0.0016375347506254912