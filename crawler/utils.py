import csv
import re
import random


def preprocess_string(s):
    # Convert to lowercase
    s = s.lower()

    # Replace common abbreviations
    abbreviations = {
        'w/': 'with',
        'w/o': 'without',
        # Add more abbreviations as needed
    }
    for abbr, full in abbreviations.items():
        s = s.replace(abbr, full)

    # Remove symbols
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)

    return s


def intersection(s1, s2):
    set1 = set(s1.lower())
    set2 = set(s2.lower())
    intersection = set1.intersection(set2)

    return intersection


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def compare_similarity(str1, str2, tolerance=0.8):
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)

    # if string is empty, assume true (not enough info, can't judge)
    if not str1 or not str2:
        return True
    intersect = intersection(str1, str2)

    str1_comp = jaccard_similarity(intersect, set(str1))
    str2_comp = jaccard_similarity(intersect, set(str2))

    return True if str1_comp > tolerance or str2_comp > tolerance else False


def get_random_sites(csv_file_path="resource/majestic_million.csv", RANDOM_SITES_COUNT=2000, MAX_ROWS_TO_READ=10000):
    random_sites = []
    with open(csv_file_path, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        random_rows = random.sample(list(csv_reader), min(RANDOM_SITES_COUNT, MAX_ROWS_TO_READ))
        for row in random_rows:
            domain = row[2]
            random_sites.append(domain)
    return random_sites


if __name__ == "__main__":
    print(compare_similarity("hello@soko.cx", "mailto:hello@soko.cx", 0.8))
