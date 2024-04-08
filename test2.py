import re

text = "sail in the boating store has it's best sail ever"

print(re.search(r"(\w+)\s+in\s+(.+)", text))
