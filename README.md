# Real-Time Object Color Identifier for Colorblind Individuals

CSC 126 - Computer Vision - Midterm Project

A Python + OpenCV desktop app that reads your laptop webcam, samples a
**Sampling Zone** in the centre of the frame, and tells you in plain words which
colours are on the object inside it - for example *"Navy Blue 62%, Red 25%,
White 13%"* - together with each colour's RGB and HEX value.

It is designed as an assistive tool for people with colour vision deficiency
(e.g. red-green colour blindness) who need a reliable colour name rather than a
guess.

## Features

- Live webcam loop with a clearly marked centre Sampling Zone
- Detects the **dominant colour and all secondary colours** in the zone
- Exact **percentage distribution** of each colour (always sums to 100%)
- Colour **name + RGB + HEX** for each colour, in a side analytics panel
- Perceptual colour matching in CIE-LAB against a 37-colour named lookup table
- Optional **object isolation** via background subtraction (segmentation mask)
- Optional gray-world **white balance** and **glare filtering** for tricky lighting
- Temporal smoothing so labels do not flicker
- Screenshot + CSV result logging for testing and the report
- Survives camera hiccups and processing errors without crashing

## Computer Vision techniques (midterm topics)

| Topic | Where it is used |
|-------|------------------|
| a. Image operations | BGR to HSV and BGR to LAB conversion, median + Gaussian blur, white balance |
| b. Image segmentation | Centre ROI crop, HSV glare threshold, background subtraction + morphology |
| d. Image analysis and transformation | K-Means clustering in LAB, Euclidean colour distance to a named palette, live bounding overlay and text labels |

## Requirements

- Python 3.9 or newer
- A working webcam (built-in laptop camera is fine)
- Windows, macOS, or Linux

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/colorblind-color-identifier.git
cd colorblind-color-identifier

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to run

```bash
python color_identifier.py
```

Hold an object so it fills the yellow-cornered **Sampling Zone**. The name and
share of the dominant colour appear under the box; the full breakdown appears in
the right-hand panel.

Optional arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `--source FILE` | none | Use an image (`.png/.jpg`) or recorded video instead of the webcam (backup for demos) |
| `--camera N` | 0 | Camera index (try 1 if you have an external webcam) |
| `--width / --height` | 960 / 540 | Requested capture resolution |
| `--k N` | 4 | Number of K-Means clusters (2-6) |
| `--zone F` | 0.30 | Sampling Zone size as a fraction of the frame |
| `--every N` | 2 | Analyse every Nth frame (raise to speed up slow laptops) |
| `--alpha F` | 0.25 | Smoothing strength (lower = steadier) |
| `--no-mirror` | off | Disable the mirrored preview |

Example: `python color_identifier.py --camera 1 --k 5 --every 3`

No webcam, or want a repeatable test? Run it on the sample inputs:

```bash
python color_identifier.py --source samples/sample_red_white_blue.png
python color_identifier.py --source samples/sample_olive_navy.png
```

The same keys work (`b`, `w`, `k`, `s`, ...). Put the object in the middle of
the picture so it falls inside the Sampling Zone.

## Keyboard controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |
| `SPACE` | Freeze / unfreeze the frame (good for screenshots) |
| `s` | Save a screenshot to `screenshots/` and log the result to `results_log.csv` |
| `b` | Capture the empty background (turns object isolation ON) |
| `c` | Clear the background (analyse the whole zone again) |
| `w` | Toggle gray-world white balance |
| `g` | Toggle glare filtering |
| `k` | Cycle the number of K-Means clusters (2-6) |
| `[` / `]` | Shrink / grow the Sampling Zone |
| `h` | Show / hide the on-screen key help |

### Tip: isolating an object from a busy background

1. Point the camera at the scene **without** the object in the zone.
2. Press `b` to capture the background.
3. Place the object in the zone. Only pixels that changed are analysed; the
   "Segmentation mask" thumbnail shows exactly what is being counted.
4. Keep the camera still after pressing `b`.

## Project structure

```
colorblind-color-identifier/
|-- color_identifier.py      # main application
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- samples/                 # sample input images (add your own photos/videos)
|-- docs/
|   |-- PROJECT_GUIDE.md     # team plan, rubric mapping, report template
|   `-- report/              # final report PDF goes here
`-- screenshots/             # test captures + results_log.csv
```

## Known limitations

- Colour names depend on lighting: warm or dim light shifts results (use `w`).
- Shiny or glossy surfaces create glare that looks white (use `g`).
- Webcams apply auto white balance and auto exposure, which can drift colours.
- The named palette has 37 colours; unusual shades snap to the nearest name.
- The tool reports colours on an object, not colour-vision-corrected images.

## Team

| Member | Role |
|--------|------|
| Member 1 | Core algorithm and processing engineer |
| Member 2 | Integration and UI/UX engineer |
| Member 3 | DevOps, documentation and QA specialist |

## License

Academic project for CSC 126. Add a license here if you plan to publish it.
