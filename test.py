import requests

# Define the URL of your API endpoint
api_url = 'http://127.0.0.1:5000/detect_memory'

# Path to the image file you want to test
image_path = "C:/Users/Michael.A_Sydani/Desktop/task3/no_memory/out15.png"

# Open the image file and read its content as binary
with open(image_path, 'rb') as img_file:
    # Prepare the request with the image file
    files = {'image': img_file}
    
    # Send the POST request to the API endpoint
    response = requests.post(api_url, files=files)

# Check if the request was successful
if response.status_code == 200:
    # Save the response content (encoded image) as a file
    with open('detected_memory.png', 'wb') as output_img:
        output_img.write(response.content)
else:
    print('Error:', response.text)
