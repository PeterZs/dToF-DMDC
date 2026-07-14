# Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors
#### [Paper](https://kaist-vclab.github.io/dToF-DMDC/static/pdfs/dtof_dense_depth_main.pdf) | [Project Page](https://kaist-vclab.github.io/dToF-DMDC/) | [Models on HF]()

> [Hakyeong Kim](https://sites.google.com/view/hakyeongkim)<sup>1*</sup>, [Ruicheng Wang](https://wangrc.site/)<sup>2, 3</sup>, [Chengtang Yao](#)<sup>2</sup>, [Jiaolong Yang](https://jlyang.org/)<sup>2</sup>, [Min H. Kim](https://vclab.kaist.ac.kr/minhkim)<sup>1</sup> <br>
> <sup>1</sup>KAIST, <sup>2</sup>Microsoft Research Asia, <sup>3</sup>USTC <br>
> In CVPR 2026 <br>

This is the official implementation of the paper **"Dense Metric Depth Completion from Sparse Direct Time-of-Flight Sensors"**, presented at **CVPR 2026**.


<img src="./assets/teaser.jpg" width="100%" alt="Method overview" align="center">


dToF-DMDC is a high-performance framework designed to generate dense metric depth maps by fusing high-resolution RGB images with sparse depth from direct Time-of-Flight (dToF) sensors. By leveraging a novel fusion architecture, our method achieves state-of-the-art results with zero-shot generalization and real-time inference speeds.


## 🚀 Key Features
* **Dense Metric Accuracy:** Accurately reconstructs continuous metric depth maps from highly sparse dToF inputs.
* **Real-Time Inference:** Runs at approximately **32 FPS** on an NVIDIA A100 GPU in half-precision (FP16), featuring a lightweight and optimized architecture.
* **Strong Zero-Shot Generalization:** Verified across **6 diverse datasets**, seamlessly adapts to unseen environments without requiring domain-specific fine-tuning.
* **RGB-dToF Fusion:**  Depth-guided dual-branch encoder with masked joint attention, enabling effective and controlled feature exchange between RGB and sparse depth.


## 📦 Code release
To be Announced