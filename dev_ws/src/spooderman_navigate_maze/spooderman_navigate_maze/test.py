import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the coordinates of the 4 points of the rectangle
min_x = -1.4643845558166504  # top-left x
max_x = 3.9821693897247314  # top-right x
min_y = -0.4697725474834442  # bottom-left y
max_y = 2.2627668380737305  # top-left y

# Calculate the width and height of the rectangle
width = max_x - min_x
height = max_y - min_y

# Calculate the size of each square along x and y axis
square_width = width / 6
square_height = height / 3

# Create the plot
fig, ax = plt.subplots()

# Draw the main rectangle (the boundary of the grid)
rect = patches.Rectangle((min_x, min_y), width, height, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(rect)

# Calculate the center of each square and add them to the plot
square_centers = []
for i in range(6):  # 6 squares along x-axis
    for j in range(3):  # 3 squares along y-axis
        # Calculate the bottom-left corner of each square
        center_x = min_x + (i + 0.5) * square_width
        center_y = max_y - (j + 0.5) * square_height
        square_centers.append((center_x, center_y, 0))  # Adding 0 for z-coordinate
        
        # Draw the square
        ax.add_patch(patches.Rectangle((min_x + i * square_width, max_y - (j + 1) * square_height), 
                                      square_width, square_height, 
                                      linewidth=1, edgecolor='blue', facecolor='lightblue', alpha=0.5))
        
        # Annotate the center of each square
        ax.text(center_x, center_y, f'({center_x:.2f}, {center_y:.2f}, 0)', ha='center', va='center', fontsize=8, color='black')

# Set the axis limits and labels
ax.set_xlim(min_x - 0.5, max_x + 0.5)
ax.set_ylim(min_y - 0.5, max_y + 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Visualization of 3x6 Grid within Rectangle')

# Show the plot
plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio is equal
plt.grid(True)
plt.show()

# Output the list of square centers in the format (x, y, 0) in the terminal
print("Coordinates of the centers of the squares:")
for index, (center_x, center_y, _) in enumerate(square_centers):
    print(f"Center of square {index + 1}: ({center_x:.2f}, {center_y:.2f}, 0)")