# OpenCV: From Pixels to Production

A beginner-to-advanced, notebook-first OpenCV course built around concepts and
real pipelines—not a catalogue of APIs. Start at the beginning, change one
parameter at a time, and keep notes on both successes and failures.

The redesign reduces the original 40 disconnected lessons to 25 focused
notebooks. Each lesson opens with motivation, learning objectives,
prerequisites, core APIs, and intuition; it closes with a mini project,
exercises, summary, best practices, pitfalls, and the next learning step.

Read [the curriculum review](CURRICULUM_REVIEW.md) for the full notebook-by-
notebook rationale, migration map, advanced-topic guidance, and missing-topic
recommendations.

## Start here

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
jupyter lab notebooks/
```

Run notebooks from either the repository root or the `notebooks/` directory.
Their setup cell locates `utils/cv_utils.py` automatically. The bundled data
is used where available; image helpers create deterministic placeholder images
for a missing image asset so an introductory example remains runnable.

## Course map

| Part | Notebooks | Outcome |
| --- | --- | --- |
| Foundations | 01–03 | Write small Python/NumPy/OpenCV pipelines and handle BGR/RGB correctly. |
| Image fundamentals | 04–06 | Read images robustly; work with pixels, masks, color, drawing, and geometry. |
| Classical image processing | 07–12 | Filter, enhance, segment, measure, and match images with clear assumptions. |
| Video | 13–15 | Process frames over time, model motion, track objects, and reason about optical flow. |
| Applied vision | 16–21 | Build document/code, face, calibration, depth, and landmark applications responsibly. |
| Deep-learning inference | 22–23 | Run ONNX models with OpenCV DNN and decode YOLO detections. |
| Engineering | 24–25 | Profile, accelerate, package, validate, and deploy a robust pipeline. |

## Notebooks

### Basic Notebooks (`01_notebooks_basic`)

| # | Notebook Name | Link |
|---|---|---|
| 00 | Getting Ready for OpenCV | [Open](01_notebooks_basic/00_Getting_Ready_for_OpenCV.ipynb) |
| 01 | Getting Started with OpenCV | [Open](01_notebooks_basic/01_Getting_Started_with_OpenCV.ipynb) |
| 02 | Image Transformations and Manipulation | [Open](01_notebooks_basic/02_Image_Transformations_and_Manipulation.ipynb) |
| 03 | Basic Image Drawing Techniques | [Open](01_notebooks_basic/03_Basic_Image_Drawing_Techniques.ipynb) |
| 04 | Working with Video and Webcam | [Open](01_notebooks_basic/04_Working_with_Video_and_Webcam.ipynb) |
| 05 | Image Filtering and Blurring | [Open](01_notebooks_basic/05_Image_Filtering_and_Blurring.ipynb) |
| 06 | Edge Detection and Thresholding | [Open](01_notebooks_basic/06_Edge_Detection_and_Thresholding.ipynb) |
| 07 | Contours and Shape Detection | [Open](01_notebooks_basic/07_Contours_and_Shape_Detection.ipynb) |
| 08 | Face and Object Detection | [Open](01_notebooks_basic/08_Face_and_Object_Detection.ipynb) |

### Advanced Course Notebooks (`02_notebooks`)

| # | Notebook Name | Link |
|---|---|---|
| 01 | Python Foundations | [Open](02_notebooks/01_Python_Foundations.ipynb) |
| 02 | NumPy for Images | [Open](02_notebooks/02_NumPy_for_Images.ipynb) |
| 03 | OpenCV Setup and First Pipeline | [Open](02_notebooks/03_OpenCV_Setup_and_First_Pipeline.ipynb) |
| 04 | Image IO Formats and Metadata | [Open](02_notebooks/04_Image_IO_Formats_and_Metadata.ipynb) |
| 05 | Pixels Channels Color and Masks | [Open](02_notebooks/05_Pixels_Channels_Color_and_Masks.ipynb) |
| 06 | Drawing and Geometric Transformations | [Open](02_notebooks/06_Drawing_and_Geometric_Transformations.ipynb) |
| 07 | Filtering and Convolution | [Open](02_notebooks/07_Filtering_and_Convolution.ipynb) |
| 08 | Contrast Histograms and Enhancement | [Open](02_notebooks/08_Contrast_Histograms_and_Enhancement.ipynb) |
| 09 | Thresholding and Morphology | [Open](02_notebooks/09_Thresholding_and_Morphology.ipynb) |
| 10 | Edges Contours and Shape Measurement | [Open](02_notebooks/10_Edges_Contours_and_Shape_Measurement.ipynb) |
| 11 | Classical Image Segmentation | [Open](02_notebooks/11_Classical_Image_Segmentation.ipynb) |
| 12 | Classical Feature and Template Matching | [Open](02_notebooks/12_Classical_Feature_and_Template_Matching.ipynb) |
| 13 | Video Processing and Background Motion | [Open](02_notebooks/13_Video_Processing_and_Background_Motion.ipynb) |
| 14 | Object Tracking | [Open](02_notebooks/14_Object_Tracking.ipynb) |
| 15 | Optical Flow | [Open](02_notebooks/15_Optical_Flow.ipynb) |
| 16 | Document and Code Vision | [Open](02_notebooks/16_Document_and_Code_Vision.ipynb) |
| 17 | Face Detection and Landmarks | [Open](02_notebooks/17_Face_Detection_and_Landmarks.ipynb) |
| 18 | Face Recognition and Responsible Use | [Open](02_notebooks/18_Face_Recognition_and_Responsible_Use.ipynb) |
| 19 | Camera Calibration and Object Pose | [Open](02_notebooks/19_Camera_Calibration_and_Object_Pose.ipynb) |
| 20 | Stereo and Depth Vision | [Open](02_notebooks/20_Stereo_and_Depth_Vision.ipynb) |
| 21 | Human Landmarks with MediaPipe | [Open](02_notebooks/21_Human_Landmarks_with_MediaPipe.ipynb) |
| 22 | OpenCV DNN and ONNX Inference | [Open](02_notebooks/22_OpenCV_DNN_and_ONNX_Inference.ipynb) |
| 23 | YOLO Object Detection | [Open](02_notebooks/23_YOLO_Object_Detection.ipynb) |
| 24 | Performance and Hardware Acceleration | [Open](02_notebooks/24_Performance_and_Hardware_Acceleration.ipynb) |
| 25 | Robust Pipelines Deployment and Review | [Open](02_notebooks/25_Robust_Pipelines_Deployment_and_Review.ipynb) |
| - | Complete Tutorial (All-in-one) | [Open](02_notebooks/__OpenCV_Complete_Tutorial.ipynb) |

## Dependencies

The core path needs Python, NumPy, Matplotlib, Jupyter, and
`opencv-contrib-python`. OCR, barcode decoding, MediaPipe, ONNX inspection,
and the deployment demo have optional dependencies listed in
`requirements.txt`. Each relevant notebook checks for optional packages and
explains the fallback rather than failing without context.

## Course conventions

- **BGR is an OpenCV storage convention.** Convert only at display/model
  boundaries, and name color-space variables clearly.
- **Preserve the source image.** Draw and experiment on copies; inspect every
  intermediate mask or transform.
- **Measure before claiming improvement.** Compare against representative data
  and an explicit downstream metric, not a single attractive screenshot.
- **Treat model outputs as uncertain.** Inspect preprocessing, tensor shape,
  confidence, failures, and boundary cases.
- **Build responsibly.** Facial and human-landmark examples require consent,
  privacy-aware data handling, representative evaluation, and no high-impact
  automated decisions.

## Repository layout

```text
notebooks/             25 lessons, in course order
data/                  small course assets and models
utils/cv_utils.py      consistent display, loading, and timing helpers
cheatsheet/            supplementary OpenCV cheat-sheet source
CURRICULUM_REVIEW.md   redesign rationale, mapping, and roadmap
```

