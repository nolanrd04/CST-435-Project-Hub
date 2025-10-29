import streamlit as st
from pathlib import Path
import os
from ner_model import HotelReviewNLPModel
from data_preprocessor import download_nltk_resources

# Download NLTK resources if not already present
download_nltk_resources()

st.set_page_config(page_title="NLP Sentiment Analysis", page_icon="🦉", layout="centered")

st.title("NLP Sentiment Analysis - TripAdvisor Hotel Reviews 🦉🏨")
st.caption("Enter a hotel review to get a sentiment prediction and visuals (requires trained model).")

# Try to load the saved model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_nlp_model.pkl")

model = None
model_loaded = False
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = HotelReviewNLPModel.load_model(MODEL_PATH)
        model_loaded = True
    except Exception as e:
        load_error = str(e)
else:
    load_error = "Trained model not found. Run ner_model.py to train and save the model."

if not model_loaded:
    st.warning(load_error)
    st.info("You can still enter text, but analysis will not run until the model is trained and saved.")
else:
    st.success("Trained model loaded — ready for analysis.")

# Create two tabs: Analyze and Docs & Visuals
tab_analyze, tab_docs = st.tabs(["Analyze", "Docs & Visuals"])

with tab_analyze:
    with st.form("review_form"):
        review_text = st.text_area("Enter hotel review text", height=150)
        submitted = st.form_submit_button("Analyze")

    if submitted:
        if not review_text.strip():
            st.error("Please enter some text to analyze.")
        elif not model_loaded:
            st.error("Model not available. Please train the model (run ner_model.py) and try again.")
        else:
            with st.spinner("Analyzing review..."):
                try:
                    result = model.predict_sentiment(review_text)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    result = None

            if result:
                st.subheader("Prediction")
                st.write(f"Processed text: {result['processed_text']}")
                st.metric("Prediction", result['prediction_label'], f"{result['positive_probability']:.2%} positive")
                st.write(f"Sentiment score (TF-IDF word-based): {result['sentiment_score']:.2f} — {result['sentiment_label']}")

                # If sentiment_scorer provides word-level scores, show them
                if 'word_scores' in result:
                    st.subheader("Top word scores")
                    try:
                        # show top 10 word scores if available
                        word_scores = result.get('word_scores', [])[:10]
                        if word_scores:
                            for w in word_scores:
                                # expecting structures like (word, score) or dict; handle common cases
                                if isinstance(w, (list, tuple)) and len(w) >= 2:
                                    st.write(f"- **{w[0]}**: {w[1]}")
                                elif isinstance(w, dict) and 'word' in w:
                                    st.write(f"- **{w['word']}**: {w.get('score', '')}")
                                else:
                                    st.write(f"- {w}")
                        else:
                            st.write("No word-level scores available for this review.")
                    except Exception:
                        st.write("Word-level scoring not available in this model build.")

                # Display saved visualizations if present (use 'width' per new API)
                viz_dir = Path(BASE_DIR) / "visualizations"
                if viz_dir.exists():
                    st.subheader("Saved Visualizations")
                    cm_path = viz_dir / "confusion_matrix.png"
                    pd_path = viz_dir / "probability_distributions.png"
                    if cm_path.exists():
                        st.image(str(cm_path), caption="Confusion Matrix", width="stretch")
                    if pd_path.exists():
                        st.image(str(pd_path), caption="Probability Distributions", width="stretch")
                else:
                    st.info("No visualizations found. Visualizations are generated when training the model (ner_model.visualize_results).")

with tab_docs:
    st.header("Documentation")
    st.markdown(
        """
        - This app uses a trained HotelReviewNLPModel (saved as `trained_nlp_model.pkl`) to predict sentiment.
        - Ratings are mapped to binary classes: 1-3 => Negative (0), 4-5 => Positive (1).
        - Text preprocessing (punctuation removal, stopword filtering) is handled by data_preprocessor.py.
        - Word-level sentiment scoring is provided by sentiment_scorer.py (TF-IDF based).
        - For text in reviews, social media, or customer feedback, sentiment analysis is suitable and effective at capturing the general emotions behind the text. This can be useful in understanding the overall meaning behind the words.
        """
    )
    st.divider()
    st.header("Saved Visualizations")
    viz_dir = Path(BASE_DIR) / "visualizations"
    if viz_dir.exists():
        cm_path = viz_dir / "confusion_matrix.png"
        pd_path = viz_dir / "probability_distributions.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", width="stretch")
        else:
            st.info("confusion_matrix.png not found in visualizations directory.")
        if pd_path.exists():
            st.image(str(pd_path), caption="Probability Distributions", width="stretch")
        else:
            st.info("probability_distributions.png not found in visualizations directory.")
    else:
        st.info("No visualizations found. Run ner_model.py to train the model and generate visuals.")
