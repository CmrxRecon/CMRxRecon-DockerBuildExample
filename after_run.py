import json

print('After running')
# TODO Send email to the submitter

with open('submitter.json', 'r') as f:
    submitter = json.load(f)

team_name = submitter['team_name']
email = submitter['email']

email = submitter['email']
print(f'Team Name: {team_name}\nEmail: {email}')