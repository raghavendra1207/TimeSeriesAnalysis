import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Pillow library for image processing
import subprocess

selected_option = None  # Global variable to store the selected option

def on_select():
    global selected_option
    selected_option = option_var.get()
    print("Selected option:", selected_option)

    # Pass the selected option to the other script
    subprocess.run(["python", "main_major.py", selected_option])

# Create the main window
window = tk.Tk()
window.title("Stock Price Tracker")

# Load and resize the background image
bg_image_path = "D:/PROGRAMS Python/majorproject/image3.png"  # Replace with the path to your new image
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((window.winfo_screenwidth(), window.winfo_screenheight()), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Set a background image
background_label = tk.Label(window, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

# Calculate the height for 5 cm (adjust the fraction as needed)
height_in_cm = 5
height_in_pixels = int((height_in_cm / 2.54) * window.winfo_fpixels("1c"))

# Create a frame to hold the label and dropdown
frame = ttk.Frame(window, style="TFrame", padding=0)
frame.place(relx=0.5, rely=0.05, anchor="n", relwidth=0.3, relheight=0.03)

# Create a style for the frame
style = ttk.Style()
style.configure("TFrame", background="")

# Create a label with a transparent background
label = ttk.Label(frame, text="Select Stock:", style="TLabel", font=("Helvetica", 18), foreground="red", background="black")
label.grid(row=0, column=0, padx=5)

# Create a variable to store the selected option
option_var = tk.StringVar()

# Create a dropdown menu (OptionMenu) with a transparent background
options = ["NIFTY 50", "Google", "Bitcoin", "FTSE100", "Tesla, Inc."]
dropdown = ttk.Combobox(frame, textvariable=option_var, values=options, style="TCombobox", height=5, font=("Helvetica", 14), background="black", foreground="black")
dropdown.grid(row=0, column=1, padx=5)
dropdown.set(options[0])  # Set the default selected option

# Bind the event handler to the dropdown menu
dropdown.bind("<<ComboboxSelected>>", on_select)

# Create a button to trigger the stock selection
style.configure("TButton", font=("Helvetica", 14))
button = ttk.Button(window, text="Select Stock", command=on_select, style="TButton", takefocus=False)
button.place(relx=0.5, rely=0.2, anchor="n", relwidth=0.2, relheight=0.05)

# Set the window size and position
window.geometry(f"{window.winfo_screenwidth()}x{window.winfo_screenheight()}+0+0")

# Run the main loop
window.mainloop()
