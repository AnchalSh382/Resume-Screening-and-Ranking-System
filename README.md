# 🧠 AI-Powered Resume Screening and Ranking System

An intelligent web application that automatically **extracts**, **analyzes**, and **ranks** resumes based on a given **job description**.  
It uses **modern NLP techniques (Sentence-Transformers)** to evaluate candidate suitability and also supports **PDF + DOCX** resume formats.

Built with **Streamlit**, this tool provides a fast and interactive way to shortlist the best candidates.

---

## 🚀 Features

### ✅ **1. Multi-format Resume Support**
- Upload multiple **PDF** or **DOCX** resumes
- Automatic text extraction using:
  - `PyPDF2` (PDF)
  - `python-docx` (DOCX)

### ✅ **2. Advanced Resume Ranking**
Two ranking methods:
1. **Sentence Transformer Embeddings (Recommended)**  
   - Uses pretrained model `all-MiniLM-L6-v2`  
   - Captures semantic meaning (e.g., *“data analysis” ≈ “analytical skills”*)

2. **TF-IDF + Cosine Similarity** (Baseline)  
   - Keyword-based matching  
   - Useful for comparison

### ✅ **3. Skill Detection**
Automatically extracts relevant skills from resumes using a predefined skill list.

### ✅ **4. Clean, Simple Web Interface**
- Built using **Streamlit**
- Upload resumes
- Enter job description
- Choose ranking method
- Click **Rank Resumes**

---

## 🛠️ Tech Stack

**Frontend & Backend:**  
- Streamlit (UI + server)

**NLP & Machine Learning:**  
- Sentence Transformers  
- Scikit-learn  
- PyTorch  

**Text Extraction:**  
- PyPDF2  
- python-docx  

---

## 📁 Project Structure

```

Resume-Screening-and-Ranking-System/
│
├── Main.py                # Main Streamlit application
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation
└── sample_resumes/        # (Optional) sample PDFs/DOCX resumes

````

---

## ▶️ How to Run the Project

### **1. Clone the repository**

```bash
git clone https://github.com/AnchalSh382/Resume-Screening-and-Ranking-System.git
cd Resume-Screening-and-Ranking-System
````

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit app**

```bash
streamlit run Main.py
```

### **4. Open the app in your browser**

Streamlit will auto-open, usually at:

```
http://localhost:8501/
```

---

## 🧪 How It Works

1. **Upload PDF/DOCX resumes**
2. **Enter job description**
3. Select ranking method:

   * *Embeddings (semantic, best)*
   * *TF-IDF (keyword)*
4. Click **Rank Resumes**
5. The app:

   * Extracts text
   * Creates embeddings or TF-IDF vectors
   * Computes similarity
   * Sorts and displays ranked candidates
   * Shows detected skills

---

## 📊 Future Enhancements

* 🔍 Extract structured fields (Name, Email, Experience, Skills)
* 🧠 Train a fine-tuned model for domain-specific ranking
* 📈 Download ranking results as CSV
* 🎨 Improve UI with themes and animations
* 📝 Add detailed resume summaries

---

## 📸 Screenshots (Optional)

> Add screenshots here:
> <img width="1920" height="1080" alt="Screenshot (540)" src="https://github.com/user-attachments/assets/f83ee829-0d8e-4829-806c-fe7f9af829f2" />
<img width="1920" height="1080" alt="Screenshot (539)" src="https://github.com/user-attachments/assets/938c25e9-1a40-468a-93ca-0e04d3c77168" />
<img width="1920" height="1080" alt="Screenshot (538)" src="https://github.com/user-attachments/assets/a490ae2d-326c-4fa1-9508-9361d56bf3d5" />


---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss what you'd like to modify.

---

## 📜 License

This project is open-source and available under the **MIT License**.

---



