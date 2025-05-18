# 🍅 Tomato Disease Classifier

A deep learning web app that classifies tomato leaf diseases into 11 distinct categories:
   * Leaf Mold
   * Tomato mosaic virus
   * powdery mildew
   * Spider mites
   * Bacterial spot
   * Early blight
   * healthy
   * Late blight
   * Tomato Yellow Leaf Curl Virus
   * Septoria leaf spot
   * Target Spot
   
* Upload an image of a tomato leaf, and the model returns the top 5 most probable diseases along with their confidence scores.

---

## 🚀 Demo

👉 **[Try the App Here](https://huggingface.co/spaces/AyoAgbaje/tomato_disease_classifier)**  
Hosted on Hugging Face Spaces using gradio.

---

## 📂 Dataset

The model was trained using a dataset from Kaggle:  
**[Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources)**  
It includes 11 classes of tomato diseases including healthy leaves.

---

## 🧠 Model Architecture

- **Backbone**: EfficientNetB0 (pretrained on ImageNet)  
- **Framework**: PyTorch  
- **Training Strategy**:  
  - Data Augmentation (Flipping, Rotation, Brightness/Contrast)
  - Transfer Learning with fine-tuning
  - Optimizer: Adam  
  - Loss Function: CrossEntropyLoss

✅ **Achieved Accuracy**: **95%**

---

## 🧪 Inference

Simply upload an image of a tomato leaf and get:
- **Top 5 predicted classes**
- **Confidence distribution for each**

Example Output:
```
1. Tomato_Yellow_Leaf_Curl_Virus: 92.1%
2. Tomato_Bacterial_Spot: 3.4%
3. Tomato_Late_Blight: 2.1%
4. Tomato_Healthy: 1.3%
5. Tomato_Leaf_Mold: 1.1%
```

---

## 📚 How to Use

1. Clone this repo:
   ```bash
   https://github.com/AgbajeAyomipo/tomato_disease_classifier.git
   cd tomato-disease-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   python run app.py
   ```

Or just click the hosted version above and try it directly in your browser!

---

## 🔗 Links

- 📁 Dataset: [Kaggle Dataset Link Here](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources)  
- 📓 Notebook: [Kaggle Notebook Link Here](https://www.kaggle.com/code/agbajeayomipo/disease-classifier-efficientnetb0-acc-95)  
- 💻 GitHub Repo: [Your GitHub Repo Link Here](https://github.com/AgbajeAyomipo/tomato_disease_classifier.git)  
- 🌐 Live Demo: [Hugging Face Spaces Link Here](https://huggingface.co/spaces/AyoAgbaje/tomato_disease_classifier)

---

## 🙌 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

MIT License