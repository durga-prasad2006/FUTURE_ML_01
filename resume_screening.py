import pandas as pd

# Load resumes
data = pd.read_csv("resumes.csv")

# Job skills
job_skills = {
    "python",
    "sql",
    "machine",
    "learning",
    "pandas",
    "numpy",
    "git"
}

results = []

for index, row in data.iterrows():

    name = row["Name"]

    resume = row["Resume"].lower().split()

    resume_skills = set(resume)

    matched = job_skills.intersection(resume_skills)

    missing = job_skills.difference(resume_skills)

    score = len(matched)

    results.append({
        "Candidate": name,
        "Score": score,
        "Matched Skills": ", ".join(sorted(matched)),
        "Missing Skills": ", ".join(sorted(missing))
    })

ranking = pd.DataFrame(results)

ranking = ranking.sort_values(by="Score", ascending=False)

print("\n===== Candidate Ranking =====")
print(ranking)

print("\n===== Best Candidate =====")
print(ranking.iloc[0])