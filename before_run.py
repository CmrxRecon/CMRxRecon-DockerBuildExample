import json

print('Before running')
# TODO load submitter information

with open('submitter.json', 'r') as f:
    submitter = json.load(f)

team_name = submitter['team_name']
email = submitter['email']
print(f'Team Name: {team_name}\nEmail: {email}')
