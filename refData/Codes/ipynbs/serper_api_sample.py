"""Http.client寫法"""
import http.client
import json

conn = http.client.HTTPSConnection("google.serper.dev")
payload = json.dumps({
  "q": "apple inc"
})
headers = {
  'X-API-KEY': '3026422e86bf19796a06b5f15433e9c0cd9bd88a',
  'Content-Type': 'application/json'
}
conn.request("POST", "/search", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))

"""request模組寫法"""
import requests
import json

url = "https://google.serper.dev/search"

payload = json.dumps({
  "q": "apple inc"
})
headers = {
  'X-API-KEY': '3026422e86bf19796a06b5f15433e9c0cd9bd88a',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)