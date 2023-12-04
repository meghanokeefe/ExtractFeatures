import csv
from content_based_features import *
import joblib
import pandas as pd

print("Enter/paste your transcript. Press Enter on an empty line to finish:")
lines = []
while True:
    line = input()
    if not line:
        break
    lines.append(line)
transcript = '\n'.join(lines)

word_list = transcript.split()  # Simple word segmentation
word_count_val = word_count(transcript)
title_word_count_val = title_word_count(transcript)  # Assuming the title is part of the transcript
document_entropy_val = compute_entropy(word_list)
easiness_val = get_readability_features(transcript)
stopword_presence_val = compute_stop_word_presence_rate(word_list)
stopword_coverage_val = compute_stop_word_coverage_rate(word_list)
preposition_rate_val = compute_preposition_rate(word_list)
auxiliary_rate_val = compute_auxiliary_verb_rate(word_list)
tobe_verb_rate_val = compute_tobe_verb_rate(word_list)
conjugate_rate_val = compute_conjunction_rate(word_list)
normalization_rate_val = compute_normalization_rate(word_list)
pronoun_rate_val = compute_pronouns_rate(word_list)
freshness_val = compute_freshness("2023-12-03")

# Dummy values for authorship and coverage topic ranks and scores
auth_topic_rank_urls_scores = [None] * 5
coverage_topic_rank_urls_scores = [None] * 5

# Dummy values for other columns
duration_val = None
speaker_speed_val = None
has_parts_val = False
type_val = None
silent_period_rate_val = None
min_engagement_val = None
max_engagement_val = None
med_engagement_val = None
mean_engagement_val = None
sd_engagement_val = None
num_learners_val = None
num_views_val = None
avg_star_rating_val = None
num_star_ratings_val = None

# Combine all features into a list
row = [
    1, None, 'misc', word_count_val, title_word_count_val,
    document_entropy_val, easiness_val, stopword_presence_val,
    stopword_coverage_val, preposition_rate_val, auxiliary_rate_val,
    tobe_verb_rate_val, conjugate_rate_val, normalization_rate_val,
    pronoun_rate_val, freshness_val, *auth_topic_rank_urls_scores,
    *coverage_topic_rank_urls_scores, duration_val, speaker_speed_val,
    has_parts_val, type_val, silent_period_rate_val, min_engagement_val,
    max_engagement_val, med_engagement_val, mean_engagement_val,
    sd_engagement_val, num_learners_val, num_views_val, avg_star_rating_val,
    num_star_ratings_val
]

# Write the features to a .csv file
with open('output1.csv', 'w', newline='') as csvfile:
    header = [
        "id", "fold", "categories", "word_count", "title_word_count", "document_entropy", "easiness",
        "fraction_stopword_presence", "fraction_stopword_coverage", "preposition_rate", "auxiliary_rate",
        "tobe_verb_rate", "conjugate_rate", "normalization_rate", "pronoun_rate", "freshness",
        "auth_topic_rank_1_url", "auth_topic_rank_1_score", "auth_topic_rank_2_url", "auth_topic_rank_2_score",
        "auth_topic_rank_3_url", "auth_topic_rank_3_score", "auth_topic_rank_4_url", "auth_topic_rank_4_score",
        "auth_topic_rank_5_url", "auth_topic_rank_5_score", "coverage_topic_rank_1_url",
        "coverage_topic_rank_1_score", "coverage_topic_rank_2_url", "coverage_topic_rank_2_score",
        "coverage_topic_rank_3_url", "coverage_topic_rank_3_score", "coverage_topic_rank_4_url",
        "coverage_topic_rank_4_score", "coverage_topic_rank_5_url", "coverage_topic_rank_5_score",
        "duration", "speaker_speed", "has_parts", "type", "silent_period_rate", "min_engagement",
        "max_engagement", "med_engagement", "mean_engagement", "sd_engagement", "num_learners",
        "num_views", "avg_star_rating", "num_star_ratings"
    ]
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerow(row)

# Read the data from the CSV file created by process_lecture_transcript
lecture_data_path = 'output1.csv'
lecture_data = pd.read_csv(lecture_data_path)
