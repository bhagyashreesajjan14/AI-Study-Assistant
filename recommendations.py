def generate_recommendations(
    performance_df
):

    if performance_df.empty:
        return []

    recommendations = []

    for _, row in performance_df.iterrows():

        score = row["average_score"]
        topic = row["topic"]

        if score < 50:

            recommendations.append({
                "topic": topic,
                "score": score,
                "priority": "HIGH",
                "recommendation":
                    "Study the fundamentals and take an easy quiz."
            })

        elif score < 75:

            recommendations.append({
                "topic": topic,
                "score": score,
                "priority": "MEDIUM",
                "recommendation":
                    "Revise the topic and practice more questions."
            })

        else:

            recommendations.append({
                "topic": topic,
                "score": score,
                "priority": "LOW",
                "recommendation":
                    "Maintain your performance and try harder questions."
            })

    priority_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2
    }

    recommendations.sort(
        key=lambda x:
        priority_order[x["priority"]]
    )

    return recommendations


def get_weak_topics(
    performance_df,
    threshold=60
):

    if performance_df.empty:
        return []

    return performance_df[
        performance_df["average_score"] < threshold
    ]["topic"].tolist()