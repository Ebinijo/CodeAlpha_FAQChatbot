import tkinter as tk
from tkinter import scrolledtext
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt', quiet=True)

# ── FAQ Data ───────────────────────────────────────────────
faqs = {
    "What is CodeAlpha?": "CodeAlpha is a leading software development company focused on innovation and emerging technologies.",
    "What services does CodeAlpha offer?": "CodeAlpha offers software development, AI solutions, web development, and internship programs.",
    "How can I apply for an internship?": "You can apply for an internship at CodeAlpha through their official website at www.codealpha.tech.",
    "What is the duration of the internship?": "The internship duration is typically 1 to 3 months depending on the domain.",
    "Will I get a certificate?": "Yes! You will receive a QR verified completion certificate after finishing the required tasks.",
    "What tasks do I need to complete?": "You need to complete a minimum of 2 or 3 tasks from your domain to be eligible for the certificate.",
    "Is the internship paid?": "The internship may be unpaid but offers certificates, recommendation letters, and placement support.",
    "What is the contact email?": "You can contact CodeAlpha at services@codealpha.tech or services.codealpha@gmail.com.",
    "What is Python?": "Python is a high-level programming language widely used in AI, data science, and web development.",
    "What is Artificial Intelligence?": "Artificial Intelligence is the simulation of human intelligence by machines to perform tasks like learning and problem solving.",
    "What is Machine Learning?": "Machine Learning is a subset of AI where systems learn from data to improve performance without being explicitly programmed.",
    "What is Deep Learning?": "Deep Learning is a subset of Machine Learning using neural networks with many layers to analyze data.",
    "What is NLP?": "Natural Language Processing (NLP) is a branch of AI that helps computers understand and interpret human language.",
    "How do I submit my task?": "Submit your completed task through the submission form shared in your WhatsApp group.",
    "What is GitHub?": "GitHub is a platform for hosting and sharing code using Git version control.",
    "How do I upload code to GitHub?": "Create a repository on GitHub, then use git commands to push your code from your local machine.",
    "What is a chatbot?": "A chatbot is an AI program that simulates conversation with users to answer questions or perform tasks.",
    "What programming languages are used in AI?": "Python, R, and Julia are commonly used in AI development.",
    "What is an API?": "An API (Application Programming Interface) allows different software applications to communicate with each other.",
    "How do I contact CodeAlpha on WhatsApp?": "You can reach CodeAlpha on WhatsApp at +91 9336576683."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# ── Vectorizer ─────────────────────────────────────────────
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(questions)

def get_answer(user_input):
    if not user_input.strip():
        return "Please type a question!"
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, vectors)
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]
    if best_score < 0.15:
        return "Sorry, I don't have an answer for that. Please contact CodeAlpha at services@codealpha.tech"
    return answers[best_idx]

# ── UI ─────────────────────────────────────────────────────
class ChatbotApp:
    def __init__(self, root):
        root.title("🤖 FAQ Chatbot")
        root.geometry("600x600")
        root.configure(bg="#1e1e2e")
        root.resizable(False, False)

        # Title
        tk.Label(root, text="🤖 FAQ Chatbot",
                 font=("Helvetica", 18, "bold"),
                 bg="#1e1e2e", fg="#cdd6f4").pack(pady=12)

        # Chat area
        self.chat_area = scrolledtext.ScrolledText(
            root, height=22, width=68,
            bg="#313244", fg="#cdd6f4",
            font=("Helvetica", 11),
            state="disabled", relief="flat",
            wrap="word"
        )
        self.chat_area.pack(padx=15, pady=5)

        # Tag colors
        self.chat_area.tag_config("user", foreground="#89b4fa")
        self.chat_area.tag_config("bot", foreground="#a6e3a1")
        self.chat_area.tag_config("label", foreground="#f5c2e7")

        # Input area
        input_frame = tk.Frame(root, bg="#1e1e2e")
        input_frame.pack(pady=10, padx=15, fill="x")

        self.input_box = tk.Entry(
            input_frame, width=48,
            bg="#313244", fg="#cdd6f4",
            font=("Helvetica", 12),
            insertbackground="white", relief="flat"
        )
        self.input_box.pack(side="left", padx=(0, 10), ipady=6)
        self.input_box.bind("<Return>", lambda e: self.send())

        tk.Button(
            input_frame, text="Send 📨",
            command=self.send,
            bg="#89b4fa", fg="#1e1e2e",
            font=("Helvetica", 11, "bold"),
            padx=14, relief="flat"
        ).pack(side="left")

        tk.Button(
            input_frame, text="Clear 🗑️",
            command=self.clear,
            bg="#f38ba8", fg="#1e1e2e",
            font=("Helvetica", 11, "bold"),
            padx=14, relief="flat"
        ).pack(side="left", padx=8)

        # Welcome message
        self.show_message("Bot", "👋 Hi! I'm FAQ Bot. Ask me anything!", "bot")

    def send(self):
        user_text = self.input_box.get().strip()
        if not user_text:
            return
        self.show_message("You", user_text, "user")
        self.input_box.delete(0, "end")
        answer = get_answer(user_text)
        self.show_message("Bot", answer, "bot")

    def show_message(self, sender, message, tag):
        self.chat_area.config(state="normal")
        self.chat_area.insert("end", f"{sender}: ", "label")
        self.chat_area.insert("end", f"{message}\n\n", tag)
        self.chat_area.config(state="disabled")
        self.chat_area.yview("end")

    def clear(self):
        self.chat_area.config(state="normal")
        self.chat_area.delete("1.0", "end")
        self.chat_area.config(state="disabled")
        self.show_message("Bot", "👋 Hi! I'm FAQ Bot. Ask me anything!", "bot")

# ── Run ────────────────────────────────────────────────────
root = tk.Tk()
app = ChatbotApp(root)
root.mainloop()