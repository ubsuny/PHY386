# How to Submit Your Final Project via GitHub Pull Request (PR) (optional)

Course Repository: github.com/ubsuny/PHY386
 
You can use a Pull Request (PR) from a fork of the course repository. 
This allows your work to be cleanly integrated into the course repo and gives me (your instructor) a clear way to review and comment on your code.

⸻

## Why Submit via Pull Request?
	•	Code Review: I can view your code, leave inline comments, and give feedback in a structured, trackable way.
	•	Professional Practice: PRs are used by developers and scientists every day when collaborating on code.
	•	Version Tracking: You can submit early and keep updating your work—GitHub will automatically update the PR.

⸻

## Instructions

1. Finish Your Project in Google Colab
	•	Complete your notebook and ensure all code runs correctly.
	•	Include titles, comments, plots, and explanations.
	•	Save it via File > Download > Download .ipynb.

2. Fork the Course Repository
	•	Visit: https://github.com/ubsuny/PHY386
	•	In the top-right, click “Fork” to create a copy in your GitHub account.

3. Create a Branch for Your Project
	•	In your forked repository, create a new branch with a short name based on your project topic.
Example:

`black-hole-modeling`


	•	You can do this directly on GitHub by clicking the branch dropdown near the top and selecting “Create branch”.

4. Add Your Notebook to the Correct Folder
	•	In your new branch, navigate to:

`2025/final/yourgithubusername/`

Replace yourgithubusername with your actual GitHub username (no spaces!).

	•	If the folder doesn’t exist yet, create it.
	•	Upload your notebook (.ipynb) into that folder.
	•	Commit the file with a descriptive message, e.g.:
`Add final project notebook – black hole modeling`

5. Open a Pull Request (PR)
	•	Go back to the main page of your fork, and you should see a “Compare & pull request” button.
If not, go to the “Pull requests” tab and choose “New pull request.”
	•	Set:
	•	Base repository: ubsuny/PHY386
	•	Base branch: main
	•	Compare: your branch from your fork (e.g., black-hole-modeling)
	•	In the PR title, write something like:
Final Project Submission – [Your Name] – [Project Title]
	•	In the PR description, give a short summary:
This notebook simulates X-ray variability in Cygnus X-1 using time-series and Fourier analysis.
	•	Click “Create pull request.”

⸻

## Important Notes
	•	You don’t have to wait until everything is perfect to open the PR.
Once the PR is created, any further commits you make to your branch in your fork will automatically update the same PR.
	•	This means you can:
	•	Submit early,
	•	Get feedback,
	•	Keep improving your work until the deadline.

⸻

## Checklist for a Good Submission
	•	Notebook is in 2025/final/yourgithubusername/
	•	All code is commented and explained
	•	Plots and outputs are included
	•	Notebook runs top to bottom without errors
	•	Your branch has a meaningful name
	•	PR title and description are clear and professional
