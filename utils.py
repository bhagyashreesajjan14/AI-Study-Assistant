def clean_text(text):

    if not text:
        return ""

    return " ".join(
        text.split()
    )


def format_subject(subject):

    return subject.replace(
        "_",
        " "
    ).title()