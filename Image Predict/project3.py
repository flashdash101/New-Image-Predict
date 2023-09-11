import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from IPython.display import Image, display as ip_display
import os
import csv
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Forest', 'Ship', 'Truck']

model = load_model('updated_imageclassifier.model')

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        first_conv_layer_name = layer.name
        break
last_conv_layer_name = first_conv_layer_name
img_size = (32, 32)
alpha = 0.4


heatmap_dir = "heatmap_images"
os.makedirs(heatmap_dir, exist_ok=True)
#Create a directory to store heatmap images



# # Create a directory to store misclassified images
# misclassified_dir = "misclassified_images"
# os.makedirs(misclassified_dir, exist_ok=True)

# Create a CSV file to store metadata
csv_filename = "misclassified_metadata.csv"
csv_file = open(csv_filename, 'a', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Correct Label", "Predicted Label", "Image Filename"])

def save_misclassified_image(image, correct_label, predicted_label):
    global misclassified_dir
    image_filename = f"misclassified_{correct_label}_{predicted_label}.jpg"
    image_path = os.path.join(misclassified_dir, image_filename)

    # Save misclassified image
    cv2.imwrite(image_path, image)

    # Record metadata in CSV file
    csv_writer.writerow([correct_label, predicted_label, image_filename])
    print(correct_label)


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = np.squeeze(jet_heatmap)
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on the original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    ip_display(Image(cam_path))

def preprocess_image(image):
    # Preprocess the image
    img = cv2.resize(image, (32, 32)) # Resize the image to 32x32 pixels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the color space from BGR to RGB
    img = np.expand_dims(img, axis=0) # Expand the dimensions of the image array
    img = img / 255.0 # Normalize the pixel values of the image
    return img

def generate_cam(image, model, class_index, layer_name):
    # Create a model that outputs the activations of the specified layer and the predictions of the original model
    grad_model = keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        # Compute the activations of the specified layer and the predictions of the original model for the input image
        conv_output, predictions = grad_model(preprocess_image(image))
        class_output = predictions[:, class_index] # Get the output for the specified class

        # Compute the gradients of the class output with respect to the activations of the specified layer
        grads = tape.gradient(class_output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # Average the gradients over all spatial locations

        # Compute a weighted sum of the activations of the specified layer using the averaged gradients as weights
        weighted_sum = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

        # Compute a heatmap by taking the maximum value over all channels and normalizing it to a range [0, 1]
        heatmap = tf.maximum(weighted_sum, 0)
        heatmap /= tf.reduce_max(heatmap)
    
        return heatmap.numpy()

def generate_and_save_cam(img_path, model, class_index, layer_name, heatmap_dir, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = generate_cam(img, model, class_index, layer_name)
    
    # Generate a unique filename for the heatmap image
    heatmap_filename = f"heatmap{len(os.listdir(heatmap_dir)) + 1}.jpg"
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)

    save_and_display_gradcam(img_path, heatmap, heatmap_path, alpha)





def select_images():
    file_paths = filedialog.askopenfilenames() # Open a file dialog to select images

    if file_paths:
        for file_path in file_paths:
            img = cv2.imread(file_path) # Read each selected image

            prediction = model.predict(preprocess_image(img)) # Make a prediction for each selected image using your model

            top_indices = np.argsort(prediction[0])[::-1][:3]  # Get indices of top 3 predicted classes
            top_probabilities = prediction[0][top_indices]     # Get probabilities of top 3 classes

            # heatmap = generate_cam(img, model, top_indices[0], last_conv_layer_name) # Generate a CAM for each selected image using your model and its last convolutional layer

            # cam_path = "cam.jpg"
            generate_and_save_cam(file_path, model, top_indices[0], last_conv_layer_name,heatmap_dir)
            #generate the CAM for each selected image using your model and its first convolutional layer

            prediction_window = tk.Toplevel(root)
            prediction_window.title('Predictions')

            for i, index in enumerate(top_indices):
                individual_accuracy = top_probabilities[i]
                rounded_accuracy = "{:.2f}".format(individual_accuracy * 100)
                #Looping through the top 3 predicted classes and displaying their probabilities
                prediction_text = 'Prediction {} is {} with a confidence of: {}%'.format(
                    i + 1, class_names[index], rounded_accuracy)
                #Print the top 3 predicted classes and their probabilities(confidence)

                prediction_label = tk.Label(prediction_window, text=prediction_text)
                prediction_label.pack()

            # # Create a new window to get user feedback
            # feedback_window = tk.Toplevel(root)
            # feedback_window.title('Feedback')

            # # Create a StringVar to store the user's feedback
            # feedback = tk.StringVar(feedback_window)
            # feedback.set('Right') # Set the default value to 'Right'

            # # Create radio buttons for the user to select if the model was right or wrong
            # right_button = tk.Radiobutton(feedback_window, text='Right', variable=feedback, value='Right')
            # right_button.pack()
            # wrong_button = tk.Radiobutton(feedback_window, text='Wrong', variable=feedback, value='Wrong')
            # wrong_button.pack()

            # # Create a button to confirm the selection
            # confirm_button = tk.Button(feedback_window, text="Confirm", command=feedback_window.destroy)
            # confirm_button.pack()

            # # Wait for the user to confirm their selection
            # feedback_window.wait_window()

            # # Check if the user selected 'Wrong'
            # if feedback.get() == 'Wrong':
            #     # Create a new window to select the correct label
            #     label_window = tk.Toplevel(root)
            #     label_window.title('Select Correct Label')

            #     # Create a StringVar to store the selected label
            #     selected_label = tk.StringVar(label_window)
            #     selected_label.set(class_names[0]) # Set the default value to the first class

            #     # Create a drop-down menu to select the label
            #     label_menu = tk.OptionMenu(label_window, selected_label, *class_names)
            #     label_menu.pack()

            #     # Create a button to confirm the selection
            #     confirm_button = tk.Button(label_window, text="Confirm", command=label_window.destroy)
            #     confirm_button.pack()

            #     # Wait for the user to confirm their selection
            #     label_window.wait_window()

            #     # Get the index of the selected label
            #     correct_index = class_names.index(selected_label.get())

            #     save_misclassified_image(img, class_names[correct_index], class_names[top_indices[0]])
            # csv_file.close()
# After displaying predictions, if user marks a prediction as incorrect:
# incorrect_index = ...  # Index of the selected prediction
# save_misclassified_image(testing_images[incorrect_index], testing_labels[incorrect_index], predicted_label)
#Possible future improvement, please don't yes or no for now.





root = tk.Tk()
root.title('Image Predictor')
button = tk.Button(root, text="Select Images", command=select_images)
button.pack()
label = tk.Label(root, text="")
label.pack()
root.mainloop()
