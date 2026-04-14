# How to Work on This GitHub Repository

This guide walks you through the typical workflow for submitting homework in PHY386 using Git and GitHub.

---

## 1. Fork & Clone (first time only)

If you haven't already, fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/PHY386.git
cd PHY386
```

Add the upstream remote so you can pull in new assignments:

```bash
git remote add upstream https://github.com/ubsuny/PHY386.git
```

---

## 2. Create a Feature Branch

Always work on a **new branch** — never commit directly to `main`.

```bash
git checkout -b <your-username>/HW5
```

A good branch name is descriptive, e.g. `jdoe/HW5` or `jdoe/fix-plot-labels`.

---

## 3. Do Your Work

Copy the homework template into your personal folder and start working:

```bash
cp 2026/HW/HW5.ipynb 2026/HW/<your-username>/HW5.ipynb
```

Open the notebook, complete the exercises, and save.

---

## 4. Stage and Commit Your Changes

```bash
git add 2026/HW/<your-username>/HW5.ipynb
git commit -m "Add HW5 solution: finite-difference heat equation"
```

Write a **descriptive** commit message — avoid vague messages like `"update"` or `"fix stuff"`.

---

## 5. Push and Open a Pull Request

```bash
git push origin <your-username>/HW5
```

Then go to [github.com/ubsuny/PHY386](https://github.com/ubsuny/PHY386), click **"Compare & pull request"**, fill in a short description, and submit.

---

## 6. Respond to Review Feedback

The instructor (or CI checks) may leave comments on your PR. Address them by pushing additional commits to the **same branch** — the PR updates automatically.

---

## Tips

| Do | Don't |
|----|-------|
| Commit early and often | Leave everything to the last minute |
| Use descriptive branch names | Work directly on `main` |
| Pull from upstream before starting | Ignore merge conflicts |
| Label axes with units in every plot | Submit notebooks with errors in cells |
