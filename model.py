# Contains model and predict function

from PIL import Image

def predict(image_path):
    # Will contain ML model
    # For now, just returns mode of image - RGB, L

    try:
        img = Image.open(image_path)
        return img.mode

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def main():
    print(predict("example.jpeg"))


if __name__ == "__main__":
    main()
