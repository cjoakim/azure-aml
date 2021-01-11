import requests
import json

# URL for the web service
scoring_uri = 'http://7155aa20-8063-4416-b886-b81a03bf6f4f.eastus2.azurecontainer.io/score'

# If the service is authenticated, set the key or token
key = ''

method = 'predict' # predict_proba or predict

# Two sets of data to score, so we get two results back
data = {"data":
        [
            [
                30.162067040804416,
                -81.35530471801758
            ]
        ],
        "method": method
        }
# Convert to JSON string
input_data = json.dumps(data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)

