import json

from agents.constants import SQL
from agents.tools import normalise_plus_combination, run_select_query

# TODO : write multi cases test data
d = [
    {"medicine": "Augmentin", "dosage": "625mg, 1-0-1 x 5 days"},
    {"medicine": "Enzoflam", "dosage": "1-0-1 x 5 days"},
    {"medicine": "Pan D", "dosage": "40mg, 1-0-0 x 5 days"},
    {"medicine": "Hexigel gum paint", "dosage": "Massage 1-0-1 x 1 week"},
]

rows = run_select_query(SQL["find_plus_combination"])
# rows1 = run_select_query(SQL["single_name"])
# print(type(rows))
# print(type(rows[0]))
# print(rows[:10])
# print("=" * 30)
# print(type(rows1))
# print(type(rows1[0]))
# print(rows1[:10])

print(rows[:10])

for name in rows[:10]:
    print(type(name[0]))

# lst = []
# for name in rows[:10]:
#     result = normalise_plus_combination(name[0])
#     lst.append(result)
#
# serialsed = json.dumps(lst)
#
# print(serialsed)
