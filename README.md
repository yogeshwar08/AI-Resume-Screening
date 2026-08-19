# AI Resume Screening & Candidate Recommendation System

An AI-powered resume screening and candidate recommendation platform that automatically analyzes multiple resumes against a target job description, calculates hybrid compatibility scores, ranks candidates, and provides explainable recruitment insights.

The system combines keyword retrieval, semantic similarity, skill matching, and candidate profile analysis to identify the most relevant candidates for a given role.

---

## Overview

Recruiters often need to evaluate large numbers of resumes against the same job description. Manually comparing candidates can be time-consuming and inconsistent.

This project automates the initial screening process by:

- Extracting text from PDF resumes
- Analyzing the target job description
- Matching resumes using keyword-based retrieval
- Measuring semantic similarity
- Evaluating technical skill compatibility
- Analyzing education, role, and experience
- Calculating an overall hybrid recommendation score
- Ranking candidates from strongest to weakest match
- Highlighting the top candidate
- Providing an interactive Streamlit interface

The goal is to provide recruiters with a faster, more consistent, and explainable first-stage candidate screening system.

---

## Key Features

### Multi-Resume Screening

Upload and analyze up to **10 PDF resumes** against a single job description.

### Hybrid Resume Matching

The recommendation engine combines multiple signals instead of relying on a single similarity metric:

- BM25 keyword matching
- Semantic similarity
- Technical skill matching
- Candidate profile matching
- Education compatibility
- Role compatibility
- Experience compatibility

### Candidate Ranking

Candidates are automatically ranked according to their final recommendation score.

Example:

```text
Rank  Candidate              Score
------------------------------------
1     Candidate A            91.6%
2     Candidate B            81.2%
3     Candidate C            63.0%
4     Candidate D            48.4%

###System Architecture


                    ┌─────────────────────┐
                    │   Job Description   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Preprocessing  │
                    └──────────┬──────────┘
                               │
                               │
       ┌───────────────────────┴────────────────────────┐
       │                                                │
       ▼                                                ▼
┌─────────────────┐                            ┌─────────────────┐
│ Candidate PDFs  │                            │ Job Description │
└────────┬────────┘                            └────────┬────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────┐                            ┌─────────────────┐
│ PDF Text        │                            │ Cleaned JD      │
│ Extraction      │                            │                 │
└────────┬────────┘                            └────────┬────────┘
         │                                              │
         └──────────────────┬───────────────────────────┘
                            │
                            ▼
                ┌─────────────────────────┐
                │ Matching Engine         │
                └────────────┬────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
        ▼                    ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌────────────────┐
│ BM25 Matching │    │ Semantic      │    │ Skill Matching │
│               │    │ Matching      │    │                │
└───────┬───────┘    └───────┬───────┘    └───────┬────────┘
        │                     │                    │
        └─────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Profile Matching    │
                    │                     │
                    │ Education           │
                    │ Role                │
                    │ Experience          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hybrid Scoring      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Candidate Ranking   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Recommendation      │
                    │ Dashboard           │
                    └─────────────────────┘
