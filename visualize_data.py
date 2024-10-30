# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load your dataset
# # Replace 'your_dataset.csv' with the path to your dataset file
# data_path = "labels.csv"  # Adjust this path as needed
# df = pd.read_csv(data_path) 

# # Display the first few rows of the dataset (optional)
# print(df.head())

# # Assuming the emotions are in a column named 'emotion'
# # You may need to adjust this based on your dataset's structure
# emotion_column = 'label'  # Change this if your emotion column has a different name

# # Count the occurrences of each emotion
# emotion_counts = df[emotion_column].value_counts()

# # Print the counts for debugging
# print("Emotion Counts:")
# print(emotion_counts)

# # Create a bar plot to visualize the distribution of emotions
# plt.figure(figsize=(10, 6))
# sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette='viridis')
# plt.title('Distribution of Emotions in Dataset')
# plt.xlabel('Emotions')
# plt.ylabel('Count')
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Show the plot
# plt.show()



# import pandas as pd

# # Load your dataset (update the path as necessary)
# data = pd.read_csv("labels.csv")

# # Filter the entries labeled as "happy"
# happy_samples = data[data['label'] == 'happy']

# # Randomly sample 10 entries (adjust the number as needed)
# sampled_happy_entries = happy_samples.sample(n=10, random_state=42)

# # Display the sampled entries
# print("Sampled entries labeled as 'happy':")
# print(sampled_happy_entries[['pth', 'label']])  # Adjust the column names as necessary




# import pandas as pd

# # Load your dataset (update the path as necessary)
# data = pd.read_csv("labels.csv")

# # Filter the entries labeled as "happy"
# happy_samples = data[data['label'] == 'happy']

# # Display the count of entries labeled as "happy"
# print(f"Total entries labeled as 'happy': {happy_samples.shape[0]}")

# # If the number of "happy" entries is small, sample a larger portion (e.g., 50)
# if happy_samples.shape[0] > 50:
#     sampled_happy_entries = happy_samples.sample(n=50, random_state=42)
# else:
#     sampled_happy_entries = happy_samples

# # Display the sampled entries
# print("Sampled entries labeled as 'happy':")
# print(sampled_happy_entries[['pth', 'label']])  # Adjust the column names as necessary










import os
import matplotlib.pyplot as plt

def visualize_data(data_dir):
    emotion_labels = os.listdir(data_dir)
    emotion_counts = {}

    # Count samples for each emotion
    for emotion in emotion_labels:
        emotion_path = os.path.join(data_dir, emotion)
        emotion_counts[emotion] = len(os.listdir(emotion_path))

    # Plotting the distribution
    plt.figure(figsize=(10, 5))
    plt.bar(emotion_counts.keys(), emotion_counts.values())
    plt.title('Emotion Distribution in FER Dataset')
    plt.xlabel('Emotions')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.show()

# Visualize training and test data
if __name__ == '__main__':
    train_dir = 'data/train1'
    test_dir = 'data/test1'
    visualize_data(train_dir)
    visualize_data(test_dir)

