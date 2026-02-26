# HW3 Experiment Selection (First Come, First Served)

## How to Claim Your Experiment

**Each student will perform a different experiment using a specific smartphone sensor.**

### Step 1: Choose an Available Experiment

Look at the table below and pick an **unclaimed** experiment.

### Step 2: Open Your Own Issue to Track Your Work

1. Go to the [Issues tab](https://github.com/ubsuny/PHY386/issues)
2. Click **"New issue"**
3. Select the **"HW3 Experiment Claim"** template
4. Fill in:
   - Title: `HW3 - [Experiment Name] - @yourusername`
   - Select your experiment from the dropdown
   - Add any initial questions or notes
5. Submit the issue

### Step 3: Comment Below to Reserve

Once you've opened your tracking issue, **comment below** with:

```
I'm claiming [Experiment Name]. Tracking in issue #[your-issue-number]
```

This reserves your experiment and lets others know it's taken.

---

## Available Experiments

| Experiment | Sensor | Physics Model | Status |
|------------|--------|---------------|--------|
| 🛗 **Elevator Pressure** | Pressure | Barometric formula: $P(h) = P_0 e^{-h/H}$ | Available |
| 🪑 **Rotating Chair** | Gyroscope | Angular damping: $\omega(t) = \omega_0 e^{-\beta t}$ | Available |
| 🧲 **Magnet Distance** | Magnetometer | Dipole field: $B(r) \propto 1/r^3$ | Available |
| 🔊 **Speed of Sound** | Microphone | Distance-time analysis (group project, 2 students) | Available |

---

## Experiment Details

### 🛗 Elevator Pressure
- **Sensor:** Pressure (barometer) — [🎥 YouTube Demo](https://www.youtube.com/watch?v=y-goBtfuXAM) | [💬 Forum Discussion](https://phyphox.org/forums/showthread.php?tid=1605)
- **What to do:** Use raw "Pressure" sensor in elevator, export CSV with time vs pressure
- **Model:** Barometric formula $P(h) = P_0 e^{-h/H}$
- **Fit parameters:** $P_0$ (sea-level pressure), $v$ (elevator speed)
- **Note:** Use **"Pressure"** sensor (not pre-made "Elevator" experiment); pressure may drift on first startup; beware double differentiation amplifies noise

### 🪑 Rotating Chair
- **Sensor:** Gyroscope — [🎥 YouTube Demo (Salad Spinner)](https://www.youtube.com/watch?v=lLCf05Hc83Y) | [💬 Forum Discussion](https://phyphox.org/forums/showthread.php?tid=409)
- **What to do:** Use raw "Gyroscope" sensor in spinning chair, export CSV with time vs angular velocity
- **Model:** Angular damping $\omega(t) = \omega_0 e^{-\beta t}$
- **Fit parameters:** $\omega_0$ (initial angular velocity), $\beta$ (damping constant)
- **Note:** Use **"Gyroscope"** sensor (not pre-made experiments); measures in rad/s; spin ~2 revolutions then let friction slow you down; video shows salad spinner but chair works same way

### 🧲 Magnet Distance
- **Sensor:** Magnetometer — [🎥 YouTube Demo (Ruler)](https://www.youtube.com/watch?v=TS0zw1ecy6A) | [💬 Forum: Measuring Field Strength](https://phyphox.org/forums/showthread.php?tid=485)
- **What to do:** Use raw "Magnetometer" sensor, measure baseline without magnet, then slowly approach magnet and subtract baseline
- **Model:** Dipole field $B(r) = A/r^3$
- **Fit parameters:** $A$ (dipole strength), $v$ (approach speed)
- **Note:** Use **uncalibrated "Magnetometer"** sensor (no auto-recalibration); find sensor location on phone; keep <2mT to avoid saturation; measure baseline first and subtract; **avoid strong fields**; video shows ruler setup, you'll move magnet toward phone instead

### 🔊 Speed of Sound (Group Project, 2 students)
- **Sensor:** Microphone (audio amplitude) — [🎥 YouTube Demo (Speed of Sound)](https://www.youtube.com/watch?v=uoUm34CnHdE) | [💬 Forum: Noise Intensity](https://phyphox.org/forums/showthread.php?tid=803)
- **What to do:** Work in pairs with two phones to measure time delay between sound creation and detection at known distance
- **Model:** Distance-time analysis $v = d/t$
- **Fit parameters:** Speed of sound $v$, initial delay $t_0$
- **Note:** Requires two students with phones; use "Acoustic Stopwatch" experiment or raw microphone; measure multiple distances; compare result to theoretical ~343 m/s at room temperature

---

## Rules

1. ✅ **First come, first served** — claim by commenting below
2. ✅ **One experiment per student** — choose carefully! (Max 2 students per experiment)
3. ✅ **Speed of Sound is a group project** — requires exactly 2 students working together
4. ✅ **Open your tracking issue first** — then comment here to reserve

---

**Once you've claimed your experiment, head to the [HW3 notebook](https://github.com/ubsuny/PHY386/blob/Homework2026/2026/HW/HW3.ipynb) to start!**
