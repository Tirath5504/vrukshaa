# Contains model and predict function

from PIL import Image
import requests
from io import BytesIO


def predict(image_url):
    # Will contain ML model
    # For now, just returns mode of image - RGB, L

    try:
        response = requests.get(image_url)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.show()

        else:
            print(f"Error: Unable to fetch image from URL. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error processing image: {e}")


def main():
    predict("https://richmondmagazine.com/downloads/25689/download/Eat%26Drink_Ingredient_Corn_GETTY_BERGAMONT_rp0719.jpg?cb=a563e4ce774a29c50f7edaebda4efaa0&w=640&h=")


if __name__ == "__main__":
    main()
