import cv2

image_path = '../../DATA/00-puppy.jpg'
image_target = cv2.imread(image_path)

while True:
    cv2.imshow('Puppy', image_target)
    
    # If you're waited at least 1 ms AND(&) we've pressed the Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()