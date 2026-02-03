# app.py - NO MATPLOTLIB VERSION
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
    }
    .prediction-ham {
        padding: 1rem;
        background-color: #D1FAE5;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
    }
    .prediction-spam {
        padding: 1rem;
        background-color: #FEE2E2;
        border-radius: 0.5rem;
        border-left: 5px solid #EF4444;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .data-card {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stat-box {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class SMSClassifier:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.pipeline = None
    
    def load_data(self):
        """Load the SMS dataset"""
        try:
            # Try to load from the uploaded file
            df = pd.read_csv('spam.csv')
            # The data has multiple empty columns, clean it
            df = df.iloc[:, :2]
            df.columns = ['label', 'message']
            df = df.dropna()
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
            return df
        except Exception as e:
            # Fallback to sample data if file not found
            st.warning(f"Using sample data. Original error: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample SMS data for demonstration"""
        data = {
            "label": ["ham", "ham", "spam", "spam", "ham", "spam", "ham", "ham", "spam", "ham",
                     "spam", "ham", "spam", "ham", "spam"],
            "message": [
                "Go until jurong point, crazy.. Available only in bugis n great world la e buffet...",
                "Ok lar... Joking wif u oni...",
                "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
                "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
                "U dun say so early hor... U c already then say...",
                "Nah I dont think he goes to usf, he lives around here though",
                "FreeMsg Hey there darling its been 3 weeks now and no word back!",
                "Even my brother is not like to speak with me. They treat me like aids patent.",
                "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles",
                "Im gonna be home soon and i dont want to talk about this stuff anymore tonight",
                "SIX chances to win CASH! From 100 to 20,000 pounds",
                "URGENT! You have won a 1 week FREE membership",
                "Congrats! 1 year special cinema pass for 2 is yours",
                "Sorry, I'll call later in meeting",
                "Todays Voda numbers ending 7548 are selected to receive a $350 award"
            ]
        }
        df = pd.DataFrame(data)
        df["label"] = df["label"].map({"ham": 0, "spam": 1})
        return df
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def preprocess_text(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        return " ".join(tokens)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        df = df.copy()
        df["cleaned"] = df["message"].apply(self.clean_text)
        df["processed"] = df["cleaned"].apply(self.preprocess_text)
        
        X = df["processed"]
        y = df["label"]
        return X, y, df
    
    def train_model(self, X_train, y_train):
        """Train the logistic regression model"""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=1000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )),
            ("classifier", LogisticRegression(
                max_iter=500,
                random_state=42,
                class_weight="balanced"
            ))
        ])
        
        self.pipeline.fit(X_train, y_train)
        return self.pipeline
    
    def predict_single(self, text):
        """Predict if a single message is spam or ham"""
        if self.pipeline is None:
            return None
        
        cleaned_text = self.clean_text(text)
        processed_text = self.preprocess_text(cleaned_text)
        
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        return {
            "prediction": "SPAM" if prediction == 1 else "HAM",
            "probability": probability[1] if prediction == 1 else probability[0],
            "confidence": max(probability) * 100
        }

def main():
    # Initialize classifier
    classifier = SMSClassifier()
    
    # Title
    st.markdown('<h1 class="main-header">📱 SMS Spam Classifier with Logistic Regression</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Home", "Data Overview", "EDA", "Model Training", "Live Prediction"]
    )
    
    # Load data
    if 'df' not in st.session_state:
        df = classifier.load_data()
        st.session_state.df = df
    else:
        df = st.session_state.df
    
    # Home Page
    if app_mode == "Home":
        st.markdown("""
        ## Welcome to the SMS Spam Classifier!
        
        This application uses **Logistic Regression** to classify SMS messages as either **SPAM** or **HAM** (not spam).
        
        ### Features:
        
        1. **Data Overview**: View the dataset structure and statistics
        2. **Exploratory Data Analysis**: Analyze data patterns with text-based reports
        3. **Model Training**: Train and evaluate the logistic regression model
        4. **Live Prediction**: Test the model with your own messages
        
        ### How it works:
        
        - The model uses **TF-IDF vectorization** to convert text to numerical features
        - **Logistic Regression** is trained to classify messages
        - Text preprocessing includes cleaning, tokenization, and lemmatization
        
        **Navigate using the sidebar to explore different sections!**
        """)
        
        st.success(f"✅ Dataset loaded successfully with {len(df)} messages")
        st.info("💡 Tip: Start by exploring the Data Overview section!")
    
    # Data Overview Page
    elif app_mode == "Data Overview":
        st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Info")
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Total messages", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Ham messages", len(df[df['label'] == 0]))
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Spam messages", len(df[df['label'] == 1]))
            st.markdown('</div>', unsafe_allow_html=True)
            
            spam_percentage = len(df[df['label'] == 1])/len(df)*100
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Spam percentage", f"{spam_percentage:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Sample Messages")
            
            tab1, tab2 = st.tabs(["Ham Samples", "Spam Samples"])
            
            with tab1:
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                ham_samples = df[df["label"] == 0]["message"].head(5).tolist()
                for i, msg in enumerate(ham_samples, 1):
                    st.write(f"**{i}.** {msg[:80]}..." if len(msg) > 80 else f"**{i}.** {msg}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                spam_samples = df[df["label"] == 1]["message"].head(5).tolist()
                for i, msg in enumerate(spam_samples, 1):
                    st.write(f"**{i}.** {msg[:80]}..." if len(msg) > 80 else f"**{i}.** {msg}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(10))
        
        # Show data types
        with st.expander("Show Data Information"):
            st.write("**Data Types:**")
            st.write(df.dtypes)
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
    
    # EDA Page (Text-based without plots)
    elif app_mode == "EDA":
        st.markdown('<h2 class="sub-header">📈 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Prepare data for EDA
        df_eda = df.copy()
        df_eda["message_length"] = df_eda["message"].apply(len)
        df_eda["word_count"] = df_eda["message"].apply(lambda x: len(str(x).split()))
        df_eda["label_name"] = df_eda["label"].map({0: "Ham", 1: "Spam"})
        
        # Statistics using text and metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Message Length Statistics")
            ham_lengths = df_eda[df_eda["label"] == 0]["message_length"]
            spam_lengths = df_eda[df_eda["label"] == 1]["message_length"]
            
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.write("**Ham Messages:**")
            st.write(f"- Average length: {ham_lengths.mean():.1f} characters")
            st.write(f"- Minimum length: {ham_lengths.min()} characters")
            st.write(f"- Maximum length: {ham_lengths.max()} characters")
            st.write(f"- Standard deviation: {ham_lengths.std():.1f} characters")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.write("**Spam Messages:**")
            st.write(f"- Average length: {spam_lengths.mean():.1f} characters")
            st.write(f"- Minimum length: {spam_lengths.min()} characters")
            st.write(f"- Maximum length: {spam_lengths.max()} characters")
            st.write(f"- Standard deviation: {spam_lengths.std():.1f} characters")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Word Count Statistics")
            ham_words = df_eda[df_eda["label"] == 0]["word_count"]
            spam_words = df_eda[df_eda["label"] == 1]["word_count"]
            
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.write("**Ham Messages:**")
            st.write(f"- Average words: {ham_words.mean():.1f}")
            st.write(f"- Minimum words: {ham_words.min()}")
            st.write(f"- Maximum words: {ham_words.max()}")
            st.write(f"- Standard deviation: {ham_words.std():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.write("**Spam Messages:**")
            st.write(f"- Average words: {spam_words.mean():.1f}")
            st.write(f"- Minimum words: {spam_words.min()}")
            st.write(f"- Maximum words: {spam_words.max()}")
            st.write(f"- Standard deviation: {spam_words.std():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Class distribution using text
        st.markdown("#### Class Distribution")
        label_counts = df_eda["label_name"].value_counts()
        total = label_counts.sum()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Ham Messages", f"{label_counts.get('Ham', 0)}", 
                     f"{label_counts.get('Ham', 0)/total*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.metric("Spam Messages", f"{label_counts.get('Spam', 0)}", 
                     f"{label_counts.get('Spam', 0)/total*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics table
        st.markdown("#### Detailed Statistical Summary")
        stats_df = df_eda.groupby("label_name").agg({
            "message_length": ["mean", "std", "min", "max"],
            "word_count": ["mean", "std", "min", "max"]
        }).round(2)
        
        st.dataframe(stats_df)
        
        # Top words analysis (text-based)
        st.markdown("#### Common Words Analysis")
        
        # Get most common words in ham messages
        ham_text = ' '.join(df_eda[df_eda["label"] == 0]["message"].astype(str).tolist())
        ham_words_list = ham_text.lower().split()
        ham_word_freq = pd.Series(ham_words_list).value_counts().head(10)
        
        # Get most common words in spam messages  
        spam_text = ' '.join(df_eda[df_eda["label"] == 1]["message"].astype(str).tolist())
        spam_words_list = spam_text.lower().split()
        spam_word_freq = pd.Series(spam_words_list).value_counts().head(10)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("**Top 10 words in Ham messages:**")
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            for word, count in ham_word_freq.items():
                st.write(f"• {word}: {count} times")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown("**Top 10 words in Spam messages:**")
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            for word, count in spam_word_freq.items():
                st.write(f"• {word}: {count} times")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Training Page
    elif app_mode == "Model Training":
        st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
        
        # Prepare data
        X, y, _ = classifier.prepare_data(df)
        
        # Train-test split
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
        with col2:
            random_seed = st.number_input("Random Seed", 0, 100, 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size/100,
            random_state=random_seed,
            stratify=y
        )
        
        # Training button
        if st.button("🚀 Train Logistic Regression Model", use_container_width=True):
            with st.spinner("Training model... This may take a moment."):
                # Train model
                classifier.train_model(X_train, y_train)
                
                # Evaluate
                y_pred = classifier.pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Store in session state
                st.session_state.classifier = classifier
                st.session_state.results = {
                    "accuracy": accuracy,
                    "report": report,
                    "conf_matrix": conf_matrix,
                    "test_size": test_size
                }
                
                st.success("✅ Model trained successfully!")
            
            # Display results
            st.markdown("### Model Performance")
            
            # Metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Ham Precision", f"{report['0']['precision']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Spam Precision", f"{report['1']['precision']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Spam Recall", f"{report['1']['recall']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed classification report
            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))
            
            # Confusion matrix as text table
            st.markdown("#### Confusion Matrix")
            cm_df = pd.DataFrame(
                conf_matrix,
                index=['Actual Ham', 'Actual Spam'],
                columns=['Predicted Ham', 'Predicted Spam']
            )
            st.dataframe(cm_df)
            
            # Display confusion matrix analysis
            st.markdown("**Confusion Matrix Analysis:**")
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.write(f"• **True Positives (Spam correctly identified):** {conf_matrix[1][1]}")
            st.write(f"• **True Negatives (Ham correctly identified):** {conf_matrix[0][0]}")
            st.write(f"• **False Positives (Ham misclassified as Spam):** {conf_matrix[0][1]}")
            st.write(f"• **False Negatives (Spam misclassified as Ham):** {conf_matrix[1][0]}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        elif "results" in st.session_state:
            st.info(f"Model already trained with {st.session_state.results['test_size']}% test split.")
            results = st.session_state.results
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("Test Size", f"{results['test_size']}%")
            
            st.write("Click the button above to retrain with different settings.")
    
    # Live Prediction Page
    elif app_mode == "Live Prediction":
        st.markdown('<h2 class="sub-header">🔍 Live SMS Spam Prediction</h2>', unsafe_allow_html=True)
        
        # Check if model is trained
        if "classifier" not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Model Training' section.")
            
            # Quick train option
            if st.button("Quick Train (with default settings)"):
                with st.spinner("Training model with default settings..."):
                    # Load and prepare data
                    X, y, _ = classifier.prepare_data(df)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Train model
                    classifier.train_model(X_train, y_train)
                    
                    # Store in session state
                    st.session_state.classifier = classifier
                    st.success("✅ Model trained with default settings!")
        else:
            classifier = st.session_state.classifier
        
        # Prediction interface
        st.markdown("### Test the Model")
        
        # Option 1: Enter custom message
        st.markdown("#### Enter Your Own Message")
        user_message = st.text_area(
            "Type or paste an SMS message:",
            height=100,
            placeholder="E.g., 'Congratulations! You've won a free ticket to Bahamas. Call now to claim your prize!'",
            key="message_input"
        )
        
        # Option 2: Use sample messages
        st.markdown("#### Or Try Sample Messages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Sample Ham Message", use_container_width=True):
                sample_ham = "Hey, are we still meeting for lunch tomorrow at 1 PM? I'll be there!"
                st.session_state.sample_message = sample_ham
        
        with col2:
            if st.button("Sample Spam Message", use_container_width=True):
                sample_spam = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
                st.session_state.sample_message = sample_spam
        
        # Use sample message if selected
        if "sample_message" in st.session_state:
            user_message = st.text_area(
                "Type or paste an SMS message:",
                value=st.session_state.sample_message,
                height=100,
                key="sample_input"
            )
        
        # Predict button
        if user_message and st.button("🔍 Predict", use_container_width=True):
            if "classifier" in st.session_state:
                # Make prediction
                result = classifier.predict_single(user_message)
                
                # Display result
                if result["prediction"] == "SPAM":
                    st.markdown(f"""
                    <div class="prediction-spam">
                        <h3>⚠️ Prediction: SPAM</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.2f}%</p>
                        <p><strong>Probability of being spam:</strong> {result['probability']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-ham">
                        <h3>✅ Prediction: HAM (Not Spam)</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.2f}%</p>
                        <p><strong>Probability of being ham:</strong> {result['probability']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show text processing (optional)
                with st.expander("View Text Processing Steps"):
                    cleaned = classifier.clean_text(user_message)
                    processed = classifier.preprocess_text(cleaned)
                    
                    st.markdown("**Original Text:**")
                    st.write(user_message)
                    
                    st.markdown("**After Cleaning:**")
                    st.write(cleaned)
                    
                    st.markdown("**After Preprocessing (tokenized & lemmatized):**")
                    st.write(processed)
            else:
                st.error("Model not available. Please train the model first.")

if __name__ == "__main__":
    main()
