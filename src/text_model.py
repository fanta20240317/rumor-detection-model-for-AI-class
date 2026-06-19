import csv
from dataclasses import dataclass
from typing import List, Optional

# path to read the csv file

@dataclass
class Record:
    item_id: str
    text: str
    label: Optional[int] = None
    event: str = ""


def read_csv(path: str, require_label: bool = True) -> List[Record]:
    records: List[Record] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = None
            if require_label:
                label = int(row["label"])
            records.append(Record(str(row.get("id", "")), row.get("text", ""), label, str(row.get("event", ""))))
    return records

