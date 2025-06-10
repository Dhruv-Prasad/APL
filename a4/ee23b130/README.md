# Keyboard Heatmap Generator

## Usage

1. **Extract Files**:
   - Extract the `ee23b130.py` and `kbd_layout.py` files from the `.zip` submission and save them to the same directory.

2. **Run the Script**:
   - To generate the heatmap, run the following command in the terminal:
     ```bash
     python ee23b130.py
     ```
   - This will create an output image file named `keyboard_heatmap.png` in the same directory.
   -The finger travel distance will be printed out in the terminal

## Customization

- **Changing the Keyboard Layout**:
   - To test the script with a different layout, you can either modify the `kbd_layout.py` file or replace it with another layout file in the same format and with the same file name (`kbd_layout.py`).

- **Changing the Input Text**:
   - To test the script with a different text input, modify the sample text string in the `ee23b130.py` script.

## Assumptions

- The y-coordinates of the keys must have **integer values** to prevent the keys from overlapping. This is because the height of each key is assumed to be 1 unit. However, the x-coordinates can be any real number.
  - **Note**: Even if the y-coordinates are not integers, the image will still be generated, but the keys might overlap.
  
- The input string must not contain any **newlines**.