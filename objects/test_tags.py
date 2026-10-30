import csv

from objects.config import TEST_TAGS_FILE


def read_test_tags():
    """Reczny opis zdjec testowych z splits/warunki_test.csv, np. wzor = stary / nowy.

    Kazda kolumna poza "image" to jeden warunek. Puste pola sa pomijane, wiec mozna opisac tylko czesc zdjec.
    """
    if not TEST_TAGS_FILE.exists():
        return {}
    with open(TEST_TAGS_FILE, newline="") as f:
        return {
            row["image"]: {name: value.strip() for name, value in row.items() if name != "image" and value.strip()}
            for row in csv.DictReader(f)
        }
