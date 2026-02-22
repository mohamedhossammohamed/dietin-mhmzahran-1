from PIL import Image

apple = Image.new('RGB', (100, 100), color='red')
apple.save('test_apple.jpg')

coffee = Image.new('RGB', (100, 100), color='brown')
coffee.save('test_coffee.jpg')

label = Image.new('RGB', (100, 100), color='white')
label.save('test_label.jpg')
