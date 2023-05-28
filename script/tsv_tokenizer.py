import fileinput

for line in fileinput.input():
  print(f'py: {line}')