import requests

url = "https://example.com/register"

num_requests = 1000

for i in range(num_requests):
    data = {
        'username': f'user{i}',
        'password': 'example_password', 
        'email': f'user{i}@example.com'
    }

    response = requests.post(url, data=data)

    if response.status_code == 200:
        print(f"Request {i+1}: Đăng ký thành công!")
    else:
        print(f"Request {i+1}: Đăng ký thất bại! Status code {response.status_code}")
