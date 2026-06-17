# How to Push Anything to GitHub via Terminal

## First Time Setup (one-time)

```bash
# 1. Navigate to your project folder
cd path/to/your/project

# 2. Initialize Git
git init

# 3. Set your identity (required before first commit)
git config user.email "your-email@example.com"
git config user.name "YourGitHubUsername"

# 4. Add the remote repository (link to GitHub)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 5. Rename default branch to 'main' (optional, recommended)
git branch -m master main
```

## Every Time You Want to Push Changes

```bash
# 1. Check what files have changed
git status

# 2. Add files to staging area
git add -A          # add ALL changed files
# OR
git add filename.py  # add a specific file only

# 3. Commit with a message
git commit -m "Describe what you changed"

# 4. Push to GitHub
git push origin main
```

## If You Already Have a Repo (cloning first)

```bash
# Clone the repo to your machine
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Make changes, then:
cd YOUR_REPO
git add -A
git commit -m "Describe changes"
git push origin main
```

## Common Commands Cheat Sheet

| Command | What it does |
|---|---|
| `git status` | Show changed files |
| `git add -A` | Stage all changes |
| `git commit -m "msg"` | Commit staged changes |
| `git push origin main` | Push to GitHub |
| `git pull origin main` | Pull latest from GitHub |
| `git log --oneline` | Show commit history |
| `git diff` | Show exact line changes |
| `git branch -M main` | Rename branch to main |

## Notes

- Always `git pull` before `git push` if working with others to avoid conflicts.
- Write clear, short commit messages in present tense (e.g., "Add login feature").
- Use `git add filename` (not `-A`) if you only want to commit specific files.
