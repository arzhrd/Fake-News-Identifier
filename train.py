import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK data
nltk.download('stopwords')

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    return ' '.join(stemmed_words)

def load_and_prepare_data():
    """Load and prepare the dataset"""
    # Sample dataset - Replace with actual dataset loading
    # df = pd.read_csv('fake_news_dataset.csv')
    
    # Creating sample data for demonstration
    sample_data = {
        'title': [
            'COVID-19 vaccine shows 95% effectiveness in trials',
            'Scientists discover aliens living among us secretly',
            'New renewable energy breakthrough reduces costs by 50%',
            'Government planning to ban all social media platforms',
            'Study confirms Mediterranean diet improves heart health',
            'Local weather service predicts heavy rainfall this week',
            'Miracle cure discovered that can reverse aging completely',
            'University researchers develop new water purification method',
            'Celebrity claims drinking bleach cures all diseases',
            'Technology company announces breakthrough in quantum computing',
            'Anonymous sources reveal government conspiracy theory',
            'Medical journal publishes study on diabetes prevention',
            'Social media influencer promotes dangerous diet trend',
            'Environmental scientists warn about climate change impacts',
            'Unverified claims about secret alien base found',
            'Economic analysis shows growth in renewable energy sector'
        ],
        'text': [
            'Clinical trials conducted across multiple countries show the new COVID-19 vaccine demonstrates 95% effectiveness in preventing severe illness.',
            'According to unnamed sources, extraterrestrial beings have been living among humans for decades without detection.',
            'Researchers at leading universities have developed revolutionary solar panel technology that reduces production costs by half.',
            'Unconfirmed reports suggest government officials are considering legislation to shut down major social media platforms.',
            'A comprehensive study involving 10,000 participants confirms the significant health benefits of following a Mediterranean diet.',
            'Meteorologists from the national weather service have issued forecasts indicating substantial precipitation expected.',
            'A mysterious scientist claims to have found the fountain of youth that can make people live forever.',
            'Environmental engineers have created an innovative filtration system that removes 99% of contaminants from water.',
            'A famous personality posted on social media encouraging followers to consume household cleaning products.',
            'Leading tech corporation announces major advancement in quantum computing that could revolutionize data processing.',
            'Conspiracy theorists spread unsubstantiated rumors about secret government operations without credible evidence.',
            'Peer-reviewed medical research demonstrates effective strategies for preventing type 2 diabetes through lifestyle changes.',
            'Popular social media personality promotes extreme fasting regimen that medical experts consider potentially harmful.',
            'Climate researchers present compelling evidence about the accelerating effects of global warming on ecosystems.',
            'Tabloid publications circulate sensationalized stories about alleged extraterrestrial facilities with no scientific backing.',
            'Financial analysts report significant investment growth in clean energy technologies and sustainable development projects.'
        ],
        'label': [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 0 = Real, 1 = Fake
    }
    
    df = pd.DataFrame(sample_data)
    
    # Combine title and text for better features
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Preprocess the text
    df['processed_text'] = df['combined_text'].apply(preprocess_text)
    
    return df

class FakeNewsTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        self.trained_models = {}
        self.results = {}
        
    def train_and_evaluate(self, X, y):
        """Train all models and evaluate performance"""
        print("Starting training process...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # Convert text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train_tfidf, y_train)
            self.trained_models[name] = model
            
            # Make predictions
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'test_labels': y_test
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
            
        return self.results
    
    def save_model_components(self, best_model_name='Random Forest'):
        """Save the best model and vectorizer"""
        
        # Determine the best model
        if not best_model_name in self.trained_models:
            best_model_name = max(self.results.keys(), 
                                key=lambda k: self.results[k]['accuracy'])
        
        best_model = self.trained_models[best_model_name]
        
        # Save model components
        model_data = {
            'model': best_model,
            'vectorizer': self.vectorizer,
            'model_name': best_model_name
        }
        
        with open('fake_news_model.pkl', 'wb') as file:
            pickle.dump(model_data, file)
        
        print(f"\nBest model ({best_model_name}) and vectorizer saved as 'fake_news_model.pkl'")
        print(f"Model accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        return best_model_name
    
    def create_visualizations(self):
        """Create performance visualization"""
        plt.figure(figsize=(15, 10))
        
        # Accuracy comparison
        plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center')
        
        # Confusion matrices for top 2 models
        sorted_models = sorted(self.results.keys(), 
                             key=lambda k: self.results[k]['accuracy'], reverse=True)
        
        for i, model_name in enumerate(sorted_models[:2]):
            plt.subplot(2, 3, i+2)
            cm = confusion_matrix(self.results[model_name]['test_labels'], 
                                self.results[model_name]['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        # Model comparison table
        plt.subplot(2, 3, 4)
        plt.axis('tight')
        plt.axis('off')
        
        table_data = []
        for model in models:
            table_data.append([model, f"{self.results[model]['accuracy']:.4f}"])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Model', 'Accuracy'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title('Model Performance Summary')
        
        # Feature importance for Random Forest
        if 'Random Forest' in self.trained_models:
            plt.subplot(2, 3, 5)
            rf_model = self.trained_models['Random Forest']
            feature_names = self.vectorizer.get_feature_names_out()
            importances = rf_model.feature_importances_
            
            # Get top 10 features
            top_indices = np.argsort(importances)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            plt.barh(range(len(top_features)), top_importances)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Importance')
            plt.title('Top 10 Important Features (Random Forest)')
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    print("="*60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Initialize trainer
    trainer = FakeNewsTrainer()
    
    # Train and evaluate models
    results = trainer.train_and_evaluate(df['processed_text'], df['label'])
    
    # Create visualizations
    trainer.create_visualizations()
    
    # Save the best model
    best_model = trainer.save_model_components()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Best Model: {best_model}")
    print(f"Accuracy: {results[best_model]['accuracy']:.4f}")
    print("Model saved as 'fake_news_model.pkl'")
    print("Performance visualization saved as 'model_performance.png'")
    print("="*60)

if __name__ == "__main__":
    main()
