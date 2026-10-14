# CSC 126 Midterm - Project Guide
**Real-Time Object Color Identifier for Colorblind Individuals**

Team of 3 | Python + OpenCV only | Plan: Sun Oct 11 to Wed Oct 14, 2026

---

## 0. Official rubric (CSC 126 Midterm PDF) and where we stand

The rubric is scored out of 100 in four categories. Use this as your final checklist.

| Category | Points | What the PDF asks for | How we cover it |
|----------|--------|-----------------------|-----------------|
| Correct use of OpenCV techniques | 30 | At least 3 of topics a-f, correctly applied, relevant to the goal, producing correct results | Topics **a, b, d** (Section 1 below). Be ready to justify each parameter in Q&A |
| Functionality of the application | 30 | Runs smoothly, no crashes or major delays; processed frames, labels, overlays, masks visible | Live webcam loop, zone box, labels, panel, mask thumbnail, crash-proof loop. Test for FPS on the demo laptop |
| Live demonstration | 20 | 15 min demo + 5 min Q&A; app running, output shown, short explanation of how it works; stay within time | Demo plan in Section 4 |
| Clarity of documentation and code quality | 20 | Clear, organised, well-commented code; 2-4 page PDF with captioned screenshots | Section 3 template; code is sectioned and commented |

Repository must contain (per the PDF): well-commented source code, `README.md` with description, setup and run steps, `requirements.txt`, **sample input files**, and **screenshots of the output**. The repo already has `samples/` (4 synthetic test images). **Member 3: add 3-5 real photos of colourful objects to `samples/` and the best screenshots to `screenshots/` before submission.**

The PDF's report list: title and objectives, problem or use case, techniques used (brief), captioned screenshots, accuracy/behaviour/limitations, conclusion. It also asks for **key learnings** in the conclusion, so Section 3.6 includes them.

---

## 1. How the app covers the three required topics

### a. Image Operations
| Technique | Where | Why |
|-----------|-------|-----|
| Gray-world white balance (`gray_world`) | whole frame, key `w` | Removes colour casts from warm/cool room light before measuring colour |
| Median blur (3x3) then Gaussian blur (5x5) | `analyze_zone`, on the downsampled ROI | Median kills salt-and-pepper sensor noise; Gaussian smooths gradients so K-Means does not split one flat surface into noisy clusters |
| `cv2.resize(..., INTER_AREA)` to 64 px | `analyze_zone` | Averages neighbouring pixels (more noise reduction) and keeps analysis fast (about 4,000 pixels) |
| BGR to HSV | glare mask | Separates brightness (V) and saturation (S) from hue, which makes "bright and washed-out" easy to threshold |
| BGR to CIE-LAB (float32, L 0-100) | colour matching | LAB is perceptually uniform: equal numeric distance is roughly equal visual difference. RGB distance is not, so naming colours in RGB gives poor results |

Report tip: show one before/after image of the blur, and explain *why LAB instead of RGB* in two sentences. Graders like the reasoning.

### b. Image Segmentation
| Technique | Where | Result |
|-----------|-------|--------|
| ROI extraction | `frame[y1:y2, x1:x2]` using `zone_rect` | Only the centre Sampling Zone is analysed |
| Threshold mask 1 (glare) | `hsv[...,2] >= 250 & hsv[...,1] <= 25`, key `g` | Removes specular highlights that would be mis-named "White" |
| Threshold mask 2 (object vs background) | `cv2.absdiff` against a stored background, then `cv2.threshold` (28), key `b` | Binary foreground mask of the object |
| Morphological opening and closing (5x5 ellipse) | after the threshold | Opening removes speckles; closing fills small holes |
| Mask applied before clustering | `pix_lab = lab[sel]` | Only object pixels reach K-Means. The "Segmentation mask" thumbnail in the panel shows this live (use it as a report screenshot) |

### d. Image Analysis and Transformation
| Technique | Where | Result |
|-----------|-------|--------|
| K-Means clustering (`cv2.kmeans`, K-Means++ seeding, K=2-6) in LAB | `analyze_zone` | Finds the main colour groups in the zone, so secondary colours are detected |
| Euclidean distance (Delta-E 76) to a 37-colour named palette | `nearest_color_name` | `sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)`; the closest palette entry gives the name ("Olive Green", "Navy Blue") |
| Cluster merging by name | `merged` dict | Two clusters that both map to "Red" are combined, giving a true percentage per colour name |
| Percentage distribution | `ColorSmoother.get` | Shares always sum to exactly 100, colours under 5% are hidden |
| Temporal smoothing (EMA) | `ColorSmoother.update` | Stops the labels flickering between frames |
| Transformation to RGB/HEX | `bgr_to_rgb_tuple`, `bgr_to_hex` | Average colour of each cluster shown as `RGB(r, g, b)` and `#RRGGBB` |
| Live overlay | `draw_zone`, `draw_panel` | Zone rectangle with corner brackets, dominant label with swatch, side panel with bars, names, percentages, thumbnails |

---

## 2. Delegation protocol (3 members, 3 days)

**Timeline:** Day 1 = Sun Oct 11 (rest of today), Day 2 = Mon Oct 12, Day 3 = Tue Oct 13, Wed Oct 14 morning = buffer and submission. Adjust if your deadline is earlier.

**Golden rules**
- Everyone starts from the same `color_identifier.py`. It is already split into sections, and each section has one owner (below). Only the owner edits their section; everyone else sends suggestions in the group chat or as a pull request.
- Work on branches: `member1-algorithm`, `member2-ui`, `member3-docs`. Merge into `main` through pull requests. Merge at the end of Day 1 and Day 2.
- Hard freeze of the code: **Tue Oct 13, 6 PM.** After that, only bug fixes.

### Member 1 - Core Algorithm and Processing Engineer
**Owns:** sections 1 (PALETTE), 3 (`gray_world`) and 4 (`analyze_zone`, `ColorSmoother`) in `color_identifier.py`.
**Does not touch:** window, drawing and key-handling code.

| Day | Tasks | Handoff |
|-----|-------|---------|
| Day 1 | Run the script, understand every line of `analyze_zone`. Test the palette with 10 real objects (red cup, navy shirt, olive jacket, etc.). Write down every wrong name. | `palette_issues.md`: object, expected name, name shown |
| Day 2 | Fix the palette: add, remove or re-tune RGB values; try K=3/4/5 and `ANALYSIS_SIZE` 48/64/96; tune `BG_DIFF_THRESH` and `MIN_PCT`. Optional: add an HSV hue check that separates "Olive Green" from "Khaki" or "Brown". | Updated `PALETTE` and constants, pushed to `member1-algorithm`, with a one-line change log in the commit messages |
| Day 3 | Final accuracy pass using Member 3's test table. Write the technical text for report sections a, b and d (using Section 1 of this guide). | `docs/report/section_techniques.md` for Member 3 to paste |

### Member 2 - Integration and UI/UX Engineer
**Owns:** sections 5 (drawing) and 6-7 (`ColorApp`, `open_camera`, `main`).
**Does not touch:** `analyze_zone` and the palette.

| Day | Tasks | Handoff |
|-----|-------|---------|
| Day 1 | Run on every team laptop; confirm the camera opens, FPS is acceptable, and keys work. Record camera index issues (`--camera 1`). | List of laptops with FPS and camera notes |
| Day 2 | Polish the UI: text sizes for a projector, panel layout, colours of the overlay. Test failure cases on purpose: cover the camera, unplug or disable it, close the window, drag it between monitors. Optimise: raise `--every` if FPS is below 15. | Stable `main` branch that never crashes; a short "demo script" (what to hold up, in what order) |
| Day 3 | Merge Member 1's changes, run the full demo three times, add a `--help` check, and freeze the code at 6 PM. | Tagged release `v1.0` |

### Member 3 - DevOps, Documentation and QA Specialist
**Owns:** repository, `requirements.txt`, `README.md`, `docs/`, test evidence, and the report PDF.

| Day | Tasks | Handoff |
|-----|-------|---------|
| Day 1 | Create the GitHub repo (layout in README), add the team as collaborators, protect `main` (require pull requests), commit the files. Verify a clean install from scratch: new venv, `pip install -r requirements.txt`, run. Create the test table (below). | Repo URL shared with the team; empty `docs/test_table.md` |
| Day 2 | QA sessions in **5 lighting conditions** (daylight window, ceiling light, warm lamp, dim room, mixed shadow) with the same 6 objects. Save screenshots with `s`; the CSV log is created for you. Start the report: sections 1-2 (title, objectives, use-case). | `screenshots/` and `results_log.csv`; filled test table; report draft v1 |
| Day 3 | Insert screenshots and section text from Member 1; write the accuracy, limitations and conclusion sections; export the PDF (2-4 pages); proofread; final clean-install test on a different laptop. | `docs/report/CSC126_Midterm_Report.pdf` and the final repo link |

**Test table template (`docs/test_table.md`)**

| Object | True colour(s) | Lighting | Detected dominant | Detected secondary | Correct? | Notes |
|--------|----------------|----------|-------------------|--------------------|----------|-------|
| Red mug | Red | Daylight | | | | |

Accuracy = correct rows / total rows. Report it overall and per lighting condition.

### Daily check-ins (10 minutes, same time each day)
1. What did I finish? 2. What is blocking me? 3. What do I need from someone else tonight?

---

## 3. Report template (2-4 pages, export to PDF)

Suggested budget: Page 1 = sections 1-2; Page 2 = section 3; Page 3 = sections 4-5; Page 4 (optional) = section 6 and references.

### 1. Title and objectives
- Title: *Real-Time Object Color Identifier for Colorblind Individuals*
- Course, section, instructor, date, names of all three members and roles.
- Objectives (3-4 bullets), for example: build a real-time colour identifier using only a laptop webcam; detect dominant and secondary colours with exact percentages; apply image operations, segmentation and analysis techniques from the course; evaluate accuracy under different lighting.

### 2. Real-world problem and use case
- About 1 in 12 men and 1 in 200 women have some form of colour vision deficiency (state the figure with a citation).
- Everyday situations: choosing clothes, reading colour-coded charts and wires, ripe fruit, traffic or status indicators.
- How the tool helps: names the colour in words, gives the proportion of each colour, and shows RGB/HEX for precise communication.
- Why a software-only approach: no extra hardware, works on any laptop.

### 3. OpenCV techniques used
Write one short paragraph per topic and always include: what the technique is, the function used, the parameter chosen, and why.
- **(a) Image operations:** gray-world white balance, median + Gaussian blur, BGR to HSV, BGR to LAB. Explain *why LAB* in two sentences. Include a figure of raw ROI vs smoothed ROI.
- **(b) Image segmentation:** centre ROI, HSV glare threshold, `absdiff` + threshold, morphological open/close. Include a figure of the ROI and its mask.
- **(d) Image analysis and transformation:** K-Means in LAB, Euclidean distance to the palette, cluster merging, percentage computation, live overlay. Include the distance formula and one worked example (cluster centre values and the three closest palette names with their distances).
- Optional small pipeline diagram: Frame, White balance, ROI, Blur, Mask, LAB, K-Means, Name matching, Overlay.

### 4. Results and screenshots (guidelines)
Take screenshots with `s` (the saved image includes the live frame, panel, ROI and mask thumbnails). Include 4-6 images, each with a caption stating the object, the lighting and the output:
1. A single-colour object (e.g. crimson mug) - shows the basic result.
2. A multi-colour object (e.g. red/white/blue flag or box) - shows dominant plus secondary colours and percentages.
3. The same object in two lighting conditions side by side - shows lighting dependency.
4. Background subtraction ON with the segmentation mask visible - shows segmentation.
5. A failure case (glare or dim light) - shows honesty about limits.
6. (Optional) White balance off vs on.

Screenshot rules: the whole window visible, the object clearly inside the zone, no personal information in the background, same resolution for all, filenames like `fig3_lighting_warm.png`. Press SPACE to freeze before pressing `s` for the cleanest image.

### 5. Accuracy, lighting and limitations
- **Method:** state the number of objects and lighting conditions tested (for example 6 objects x 5 conditions = 30 trials) and how "correct" was judged (name within the expected colour family).
- **Results:** table of accuracy per lighting condition and overall; list the most common confusions (for example Olive Green vs Khaki, Navy Blue vs Black in dim light).
- **Lighting dependency:** colour temperature and brightness change pixel values; the camera's auto white balance and auto exposure also shift colours; white balance (`w`) helps but cannot recover everything.
- **Limitations:** finite 37-colour palette; glare on glossy surfaces; shadows split one colour into two shades; webcam colour accuracy is not scientific-grade; background subtraction needs a still camera; edges of objects blend colours; speed depends on the laptop.
- **Not claimed:** this is an assistive aid, not a medical or calibrated colorimeter.

### 6. Conclusion and future improvements
- Conclusion: 3-4 sentences restating what was built, that all three required topics are used, and the main measured accuracy.
- Key learnings (the rubric asks for these): for example why LAB beats RGB for naming colours; why blur before clustering matters; how much lighting changes results; what working in a 3-person repo taught you (branches, pull requests, ownership).
- Future work: automatic calibration using a white card or colour checker; a larger or user-editable palette loaded from CSV; adaptive K selection (elbow or silhouette); GrabCut or contour-based object segmentation; speech output of colour names; a mobile or web version; colour-vision-deficiency simulation or correction filters.

## 4. Live demo plan (15 min demo + 5 min Q&A)

Rehearse this twice on the actual presentation laptop. Stay under 15 minutes.

| Minutes | Who | What |
|---------|-----|------|
| 0-2 | Member 3 | Problem: colour blindness, who it affects, what the app does |
| 2-5 | Member 1 | Pipeline slide or whiteboard: white balance, ROI, blur, mask, LAB, K-Means, name matching. Name topics a, b, d |
| 5-11 | Member 2 | Live run: single-colour object, then a multi-colour object (point at percentages, RGB/HEX), then show `k`, `w`, `g`, `[ ]`, freeze and `s` |
| 11-13 | Member 1 | Segmentation demo: press `b` on an empty zone, add an object, show the mask thumbnail |
| 13-14 | Member 3 | Results table: accuracy per lighting condition, honest limitations |
| 14-15 | Member 2 | Wrap-up and what you would improve |
| 15-20 | All | Q&A |

Backup plan (rehearse it): if the webcam fails, run `python color_identifier.py --source samples/sample_red_white_blue.png`. Keep a short recorded video of a working run on the desktop too.

Likely Q&A questions, with one-line answers to prepare:
- Why LAB instead of RGB or HSV? Distances in LAB match human colour difference; HSV is used only for the glare threshold.
- Why K-Means and how did you choose K? It groups similar pixels without knowing the colours in advance; K=4 default, adjustable 2-6; too high splits one colour into shades, too low merges colours.
- How do you get the percentage? Pixels per cluster divided by analysed pixels, clusters with the same name merged.
- What happens in bad lighting? Names shift; white balance and glare filter help; report the measured accuracy.
- Why background subtraction and not a contour detector? It is simple and robust for a fixed camera; GrabCut or contours are listed as future work.
- What would break it? Moving camera after `b`, glossy objects, very dim rooms.

---

### Final checklist before submitting
- [ ] 2-4 pages, PDF, all member names on page 1
- [ ] Every figure has a caption and is referenced in the text
- [ ] Topics a, b and d are each clearly labelled
- [ ] GitHub link works in a private/incognito window
- [ ] Fresh clone + `pip install -r requirements.txt` + `python color_identifier.py` works on a second laptop
- [ ] Repo has README, requirements.txt, sample inputs in `samples/`, and output screenshots in `screenshots/`
- [ ] Conclusion includes key learnings and possible improvements
- [ ] Backup `--source` demo tested and a recorded video ready
- [ ] Demo rehearsed with the actual presentation laptop and lighting
