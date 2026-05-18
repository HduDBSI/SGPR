"""
Configuration file for the RAG evaluation pipeline.
Contains global settings, model configurations, and prompt templates.
"""

import os

# --- Knowledge Graph Configuration ---
KG_BASE_DIR = 'data/kg_final'

# --- LLM Configuration ---
OPENAI_MODEL_ID = 'gpt-4o-mini'

# --- API Configuration ---
OPENAI_API_BASE_URL = ""
OPENAI_API_TIMEOUT = 180

# --- Retry Configuration ---
MAX_RETRY_ATTEMPTS = 5
RETRY_MIN_WAIT = 10
RETRY_MAX_WAIT = 60

# --- LLM Prompt Templates ---

# System prompt for path-based reasoning
ICL_SYS_PROMPT_PATHS = (
    "You are an expert knowledge graph reasoner. "
    "You will be provided with a set of retrieval paths. Some paths represent direct facts (1-hop), "
    "while others are multi-hop reasoning chains. "
    "Your sole purpose is to answer user questions by analyzing these paths and extracting relevant entities. "
    "Format your answers as a list, with each answer entity on a new line, prefixed by 'ans:'."
)

# Few-shot user example for path-based questions
ICL_USER_PROMPT_PATHS = """Paths:
Path: (Emmitt Smith,people.person.education,m.02kycwv) -> (m.02kycwv,education.education.institution,University of Florida) -> (University of Florida,education.educational_institution.sports_teams,Florida Gators football) -> (Florida Gators football,sports.sports_team.fight_song,The Orange and Blue)
Path: (Emmitt Smith,people.person.education,m.02kycwv) -> (m.02kycwv,education.education.institution,University of Florida) -> (Florida Gators football,sports.school_sports_team.school,University of Florida)
Path: (Emmitt Smith,american_football.football_player.position_s,Running back) -> (Running back,sports.sports_position.sport,American football) -> (Florida Gators football,sports.sports_team.sport,American football)
Path: (Emmitt Smith,people.person.nationality,United States of America) -> (Gainesville,location.location.containedby,United States of America) -> (Florida Gators football,sports.sports_team.location,Gainesville)
Path: (University of Florida,education.educational_institution.sports_teams,Florida Gators football) -> (Florida Gators football,sports.sports_team.fight_song,The Orange and Blue)
Path: (Florida Gators football,sports.sports_team.fight_song,The Orange and Blue)
Path: (Emmitt Smith,sports.pro_athlete.teams,m.0hqf002) -> (m.0hqf002,sports.sports_team_roster.team,Florida Gators football)
Path: (Emmitt Smith,people.person.education,m.02kycwv) -> (m.02kycwv,education.education.institution,University of Florida) -> (University of Florida,education.educational_institution.mascot,Albert and Alberta Gator) -> (Florida Gators football,sports.sports_team.team_mascot,Albert and Alberta Gator)

Question:
Which school with the fight song "The Orange and Blue" did Emmitt Smith play for?"""

# Few-shot assistant example for path-based questions
ICL_ASS_PROMPT_PATHS = """To answer the question, we need to find a school that satisfies two conditions: it has "The Orange and Blue" as its fight song, and Emmitt Smith played for it.

First, let's identify which entity has "The Orange and Blue" as its fight song.
From the path `(Florida Gators football,sports.sports_team.fight_song,The Orange and Blue)`, we clearly see that Florida Gators football is the team with this fight song.

Next, we need to determine if Emmitt Smith played for the Florida Gators football team.
Several paths connect Emmitt Smith to Florida Gators football:
The path `(Emmitt Smith,people.person.education,m.02kycwv) -> (m.02kycwv,education.education.institution,University of Florida) -> (University of Florida,education.educational_institution.sports_teams,Florida Gators football)` shows Emmitt Smith's education is linked to the University of Florida, which has Florida Gators football as its sports team.
Another path `(Emmitt Smith,sports.pro_athlete.teams,m.0hqf002) -> (m.0hqf002,sports.sports_team_roster.team,Florida Gators football)` directly states Emmitt Smith played for Florida Gators football through a roster entry.

Since Florida Gators football is the team with "The Orange and Blue" fight song and Emmitt Smith played for them, it is the correct answer.
Therefore, the formatted answer is:
ans: Florida Gators football"""

# Chain-of-Thought prompt for difficult questions
ICL_COT_PROMPT = (
    "Let's think step by step to find the correct answer from the provided paths: "
    "Trace each path starting from the entity mentioned in the question. "
    "Analyze the sequence of relations and intermediate entities to see if they logically satisfy the question's constraints. "
    "Identify the final target entities (leaf nodes) at the end of valid reasoning chains. "
    "If multiple distinct entities are reached through valid paths, you MUST list ALL of them. "
    "Provide the final entities one per line with the prefix 'ans:'. "
    "If no path leads to a logical answer, return 'ans: not available'."
)

# --- Dataset Configuration ---
DATASET_REPO_MAP = {
    'webqsp': 'rmanluo/RoG-webqsp',
    'cwq': 'rmanluo/RoG-cwq'
}