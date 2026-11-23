
import streamlit as st
import torch
import torch.nn as nn
import re

st.set_page_config(
    page_title="RNN Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# Model architecture
class NextWordRNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        last_output = output[:, -1, :]
        return self.fc(last_output)


# Load model
@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "next_word_rnn.pth",
        map_location="cpu",
        weights_only=True
    )

    vocab_size = checkpoint["vocab_size"]

    model = NextWordRNN(vocab_size)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint["word_to_idx"], checkpoint["idx_to_word"]


model, word_to_idx, idx_to_word = load_model()


# Prediction function
def predict_next_word(sentence):

    words = re.findall(
        r"\b\w+\b|[^\w\s]",
        sentence.lower()
    )

    words = words[-5:]

    indices = [
        word_to_idx.get(word, word_to_idx["<UNK>"])
        for word in words
    ]

    indices = (
        [word_to_idx["<PAD>"]] * (5 - len(indices))
        + indices
    )

    x = torch.tensor([indices], dtype=torch.long)

    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)

        top_probs, top_indices = torch.topk(
            probabilities,
            5,
            dim=1
        )

    results = []

    for idx, prob in zip(
        top_indices[0].tolist(),
        top_probs[0].tolist()
    ):

        results.append({
            "word": idx_to_word[idx],
            "probability": prob * 100
        })

    return results


# Streamlit UI
st.title("🧠 Next Word Prediction")
st.subheader("Powered by PyTorch Simple RNN")

st.write(
    "Enter an English sentence and let the RNN model "
    "predict the next word."
)

sentence = st.text_input(
    "Enter your sentence:",
    placeholder="What is your favorite"
)

if st.button("Predict Next Word", type="primary"):

    if sentence.strip():

        results = predict_next_word(sentence)

        st.success("Prediction completed!")

        st.subheader("Top 5 Predictions")

        for i, result in enumerate(results):

            st.write(
                f"**{i+1}. {result['word']}**"
            )

            st.progress(
                min(result["probability"] / 100, 1.0)
            )

            st.caption(
                f"Probability: {result['probability']:.2f}%"
            )

    else:
        st.warning("Please enter a sentence.")

st.divider()

st.caption(
    "Machine Learning Project | PyTorch | Simple RNN"
)
