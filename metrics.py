import matplotlib.pyplot as plt
import numpy as np

models = ["Baseline RF", "Depth-Adaptive RF"]
train_accuracies = [96.15, 92.79]
test_accuracies = [83.15, 86.52]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
rects1 = ax.bar(x - width/2, train_accuracies, width, label='Train Accuracy')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Model')
ax.set_title('Training vs. Test Accuracy for Random Forest Models')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 100)

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}%', xy=(rect.get_x() + rect.get_width()/2, height), xytext=(0,3),  textcoords="offset points", ha='center', va='bottom')

plt.tight_layout()
plt.show()