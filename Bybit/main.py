def _calculate_factor_scores(data):
    # Function modified to evaluate both long and short scores and to reverse their directions
    long_scores = []
    short_scores = []
    for item in data:
        # Assuming `score` is extracted from item
        score = item['score']
        long_scores.append(-score)  # Reverse direction for long scores
        short_scores.append(score)  # Keep short scores as is
    return long_scores, short_scores
