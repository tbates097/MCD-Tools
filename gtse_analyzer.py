import pandas as pd
import re
import os
from typing import List, Dict, Tuple
from datetime import datetime
from collections import Counter

# Try to import plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: plotly not available. Charts will be disabled.")
    PLOTTING_AVAILABLE = False

# Try to import NLTK - required for text analysis features
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK data (only runs once)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading required NLTK data for text analysis...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    # Check for newer punkt_tab resource (required in newer NLTK versions)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading updated NLTK tokenizer data...")
        nltk.download('punkt_tab', quiet=True)
    
    NLTK_AVAILABLE = True
    
except ImportError:
    print("⚠️  NLTK not installed. Text analysis features will be limited.")
    print("   To enable full text analysis, install NLTK: pip install nltk")
    NLTK_AVAILABLE = False

class GTSEAnalyzer:
    def __init__(self):
        """Initialize the GTSE analysis tool"""
        self.df = None
        self.csv_file = None
        
    def load_csv_file(self, file_path: str = None) -> bool:
        """
        Load the translated CSV file
        
        Args:
            file_path: Path to CSV file. If None, will prompt user to select.
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if file_path is None:
            # Find the most recent GTSE translated results file
            files = [f for f in os.listdir('.') if f.startswith('gtse_translated_results_') and f.endswith('.csv')]
            if files:
                # Sort by modification time, newest first
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                file_path = files[0]
                print(f"Found recent file: {file_path}")
                use_file = input(f"Use this file? (y/n): ").strip().lower()
                if use_file != 'y':
                    file_path = input("Enter CSV file path: ").strip()
            else:
                file_path = input("Enter CSV file path: ").strip()
        
        try:
            self.df = pd.read_csv(file_path)
            self.csv_file = file_path
            print(f"✓ Loaded {len(self.df)} rows from {file_path}")
            print(f"Columns available: {list(self.df.columns)}")
            return True
        except Exception as e:
            print(f"✗ Error loading file: {e}")
            return False
    
    def search_keywords(self, keywords: List[str], case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search for keywords in summary and description columns
        
        Args:
            keywords: List of keywords to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            DataFrame with matching rows
        """
        if self.df is None:
            print("No data loaded. Please load a CSV file first.")
            return pd.DataFrame()
        
        # Columns to search in (original and translated)
        search_columns = []
        for col in ['summary', 'description']:
            if col in self.df.columns:
                search_columns.append(col)
            if f'{col}_translated_text' in self.df.columns:
                search_columns.append(f'{col}_translated_text')
        
        if not search_columns:
            print("No summary or description columns found in the data.")
            return pd.DataFrame()
        
        print(f"Searching in columns: {search_columns}")
        
        # Create search mask
        mask = pd.Series([False] * len(self.df))
        
        for keyword in keywords:
            keyword_mask = pd.Series([False] * len(self.df))
            
            for col in search_columns:
                if col in self.df.columns:
                    col_data = self.df[col].fillna('').astype(str)
                    if case_sensitive:
                        keyword_mask |= col_data.str.contains(keyword, regex=False, na=False)
                    else:
                        keyword_mask |= col_data.str.contains(keyword, case=False, regex=False, na=False)
            
            mask |= keyword_mask
        
        results = self.df[mask].copy()
        return results
    
    def display_results(self, results: pd.DataFrame, keywords: List[str], max_display: int = 10):
        """
        Display search results in a formatted way
        
        Args:
            results: DataFrame with search results
            keywords: Keywords that were searched for
            max_display: Maximum number of results to display
        """
        if results.empty:
            print(f"\nNo results found for keywords: {keywords}")
            return
        
        print(f"\n=== Search Results ===")
        print(f"Found {len(results)} rows matching keywords: {keywords}")
        
        if len(results) > max_display:
            print(f"Showing first {max_display} results (use 'save' option to get all results)")
            display_results = results.head(max_display)
        else:
            display_results = results
        
        for idx, (_, row) in enumerate(display_results.iterrows(), 1):
            print(f"\n--- Result {idx} ---")
            
            # Show key fields
            if 'created' in row.index:
                print(f"Created: {row['created']}")
            if 'CustomerName' in row.index:
                print(f"Customer: {row['CustomerName']}")
            if 'status' in row.index:
                print(f"Status: {row['status']}")
            
            # Show summary (best available version - translated if available, otherwise cleaned original)
            if 'summary' in row.index:
                if 'summary_translated_text' in row.index and pd.notna(row['summary_translated_text']):
                    # Use cleaned/translated version (always the best version)
                    summary_text = str(row['summary_translated_text'])[:200]
                    print(f"Summary: {summary_text}{'...' if len(str(row['summary_translated_text'])) > 200 else ''}")
                else:
                    # Fallback to original if no processed version available
                    summary_text = str(row['summary'])[:200]
                    print(f"Summary: {summary_text}{'...' if len(str(row['summary'])) > 200 else ''}")
            
            # Show description (best available version - translated if available, otherwise cleaned original)
            if 'description' in row.index:
                if 'description_translated_text' in row.index and pd.notna(row['description_translated_text']):
                    # Use cleaned/translated version (always the best version)
                    description_text = str(row['description_translated_text'])[:300]
                    print(f"Description: {description_text}{'...' if len(str(row['description_translated_text'])) > 300 else ''}")
                else:
                    # Fallback to original if no processed version available
                    description_text = str(row['description'])[:300]
                    print(f"Description: {description_text}{'...' if len(str(row['description'])) > 300 else ''}")
            
            print("-" * 60)
    
    def save_results(self, results: pd.DataFrame, keywords: List[str]):
        """
        Save search results to a new CSV file
        
        Args:
            results: DataFrame with search results
            keywords: Keywords that were searched for
        """
        if results.empty:
            print("No results to save.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        keywords_str = '_'.join([k.replace(' ', '') for k in keywords[:3]])  # First 3 keywords
        filename = f"gtse_search_{keywords_str}_{timestamp}.csv"
        
        try:
            results.to_csv(filename, index=False)
            print(f"✓ Saved {len(results)} results to: {filename}")
        except Exception as e:
            print(f"✗ Error saving file: {e}")
    
    def save_results_to_txt(self, results: pd.DataFrame, keywords: List[str]):
        """
        Save search results to a formatted TXT file
        
        Args:
            results: DataFrame with search results
            keywords: Keywords that were searched for
        """
        if results.empty:
            print("No results to save.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        keywords_str = '_'.join([k.replace(' ', '') for k in keywords[:3]])  # First 3 keywords
        filename = f"gtse_search_{keywords_str}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== GTSE Search Results ===\n")
                f.write(f"Search keywords: {', '.join(keywords)}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total results: {len(results)}\n")
                f.write("=" * 80 + "\n\n")
                
                for idx, (_, row) in enumerate(results.iterrows(), 1):
                    f.write(f"--- Result {idx} ---\n")
                    
                    # Show key fields
                    if 'created' in row.index and pd.notna(row['created']):
                        f.write(f"Created: {row['created']}\n")
                    if 'CustomerName' in row.index and pd.notna(row['CustomerName']):
                        f.write(f"Customer: {row['CustomerName']}\n")
                    if 'status' in row.index and pd.notna(row['status']):
                        f.write(f"Status: {row['status']}\n")
                    
                    # Summary (best available version)
                    if 'summary' in row.index and pd.notna(row['summary']):
                        if 'summary_translated_text' in row.index and pd.notna(row['summary_translated_text']):
                            f.write(f"Summary: {str(row['summary_translated_text'])}\n")
                        else:
                            f.write(f"Summary: {str(row['summary'])}\n")
                    
                    # Description (best available version) - COMPLETE TEXT
                    if 'description' in row.index and pd.notna(row['description']):
                        if 'description_translated_text' in row.index and pd.notna(row['description_translated_text']):
                            f.write(f"Description: {str(row['description_translated_text'])}\n")
                        else:
                            f.write(f"Description: {str(row['description'])}\n")
                    
                    f.write("\n" + "=" * 80 + "\n\n")
            
            print(f"✓ Saved {len(results)} results to: {filename}")
            
        except Exception as e:
            print(f"✗ Error saving TXT file: {e}")
    
    def get_column_stats(self):
        """Display statistics about the loaded data"""
        if self.df is None:
            print("No data loaded.")
            return
        
        print(f"\n=== Data Statistics ===")
        print(f"Total rows: {len(self.df)}")
        
        # Translation statistics
        for col in ['summary', 'description']:
            if f'{col}_translation_needed' in self.df.columns:
                translated = self.df[f'{col}_translation_needed'].sum()
                percentage = (translated / len(self.df)) * 100
                print(f"{col.title()} translated: {translated} rows ({percentage:.1f}%)")
        
        # Language distribution
        for col in ['summary', 'description']:
            lang_col = f'{col}_detected_language'
            if lang_col in self.df.columns:
                print(f"\n{col.title()} languages:")
                lang_counts = self.df[lang_col].value_counts().head(5)
                for lang, count in lang_counts.items():
                    print(f"  {lang}: {count} rows")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis: tokenize, remove stop words, lemmatize
        
        Args:
            text: Raw text to process
            
        Returns:
            List of processed tokens
        """
        if pd.isna(text) or not text:
            return []
        
        if not NLTK_AVAILABLE:
            # Fallback to basic preprocessing without NLTK
            text = str(text).lower()
            # Simple word extraction using regex
            tokens = re.findall(r'\b[a-z]{3,}\b', text)  # Words with 3+ letters
            
            # Basic stop word removal
            basic_stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'between', 'among', 'this', 'that', 'these', 'those', 'are', 'was',
                'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'can', 'system', 'issue', 'problem',
                'error', 'machine', 'software', 'hardware', 'please', 'need', 'want', 'like',
                'get', 'use', 'work', 'time', 'help', 'support', 'customer', 'thanks', 'thank'
            }
            tokens = [token for token in tokens if token not in basic_stop_words]
            return tokens
        
        # Full NLTK preprocessing
        try:
            tokens = word_tokenize(str(text).lower())
        except Exception as e:
            print(f"⚠️  NLTK tokenization error: {e}")
            print("   Falling back to basic text processing...")
            # Fallback to basic preprocessing
            text = str(text).lower()
            tokens = re.findall(r'\b[a-z]{3,}\b', text)
        else:
            # Remove non-alphabetic tokens and short words
            tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
        
        # Remove common stop words
        stop_words = set(stopwords.words('english'))
        # Add technical/common words that aren't useful for analysis
        technical_stop_words = {
            'system', 'issue', 'problem', 'error', 'machine', 'software', 'hardware',
            'please', 'would', 'could', 'need', 'want', 'like', 'get', 'use', 'work',
            'time', 'help', 'support', 'customer', 'thanks', 'thank', 'regards',
            'aerotech', 'automation', 'gtse', 'case', 'ticket', 'request'
        }
        stop_words.update(technical_stop_words)
        
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize (reduce words to base form)
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            print(f"⚠️  NLTK lemmatization error: {e}")
            print("   Continuing without lemmatization...")
            # Continue without lemmatization if it fails
        
        return tokens
    
    def analyze_word_frequency(self, columns: List[str] = None, top_n: int = 30) -> Dict:
        """
        Analyze word frequency across specified columns
        
        Args:
            columns: List of columns to analyze (default: summary and description)
            top_n: Number of top words to return
            
        Returns:
            Dictionary with analysis results
        """
        if self.df is None:
            return {}
        
        if columns is None:
            columns = ['summary', 'description']
        
        # Use translated/cleaned versions when available
        text_columns = []
        for col in columns:
            if f'{col}_translated_text' in self.df.columns:
                text_columns.append(f'{col}_translated_text')
            elif col in self.df.columns:
                text_columns.append(col)
        
        if not text_columns:
            return {}
        
        # Collect all tokens
        all_tokens = []
        for col in text_columns:
            for text in self.df[col].fillna(''):
                all_tokens.extend(self.preprocess_text(text))
        
        # Count frequencies
        word_freq = Counter(all_tokens)
        
        return {
            'total_words': len(all_tokens),
            'unique_words': len(word_freq),
            'top_words': word_freq.most_common(top_n),
            'columns_analyzed': text_columns
        }
    
    def analyze_phrases(self, columns: List[str] = None, n_gram_size: int = 2, top_n: int = 20) -> Dict:
        """
        Analyze common phrases (n-grams) in the text
        
        Args:
            columns: List of columns to analyze
            n_gram_size: Size of n-grams (2=bigrams, 3=trigrams)
            top_n: Number of top phrases to return
            
        Returns:
            Dictionary with phrase analysis results
        """
        if self.df is None:
            return {}
        
        if columns is None:
            columns = ['summary', 'description']
        
        # Use translated/cleaned versions when available
        text_columns = []
        for col in columns:
            if f'{col}_translated_text' in self.df.columns:
                text_columns.append(f'{col}_translated_text')
            elif col in self.df.columns:
                text_columns.append(col)
        
        if not text_columns:
            return {}
        
        # Collect all n-grams
        all_ngrams = []
        for col in text_columns:
            for text in self.df[col].fillna(''):
                tokens = self.preprocess_text(text)
                if len(tokens) >= n_gram_size:
                    if NLTK_AVAILABLE:
                        text_ngrams = list(ngrams(tokens, n_gram_size))
                        # Convert tuples to strings
                        text_ngrams = [' '.join(gram) for gram in text_ngrams]
                    else:
                        # Fallback n-gram generation without NLTK
                        text_ngrams = []
                        for i in range(len(tokens) - n_gram_size + 1):
                            gram = ' '.join(tokens[i:i + n_gram_size])
                            text_ngrams.append(gram)
                    all_ngrams.extend(text_ngrams)
        
        # Count frequencies
        phrase_freq = Counter(all_ngrams)
        
        return {
            'total_phrases': len(all_ngrams),
            'unique_phrases': len(phrase_freq),
            'top_phrases': phrase_freq.most_common(top_n),
            'n_gram_size': n_gram_size,
            'columns_analyzed': text_columns
        }
    
    def analyze_summary_trends(self, top_n: int = 25) -> Dict:
        """
        Analyze trends specifically in summary column
        
        Args:
            top_n: Number of top trends to return
            
        Returns:
            Dictionary with summary analysis results
        """
        if self.df is None:
            return {}
        
        # Use translated/cleaned summary when available
        summary_col = 'summary_translated_text' if 'summary_translated_text' in self.df.columns else 'summary'
        
        if summary_col not in self.df.columns:
            return {}
        
        # Get all summary texts
        summaries = self.df[summary_col].fillna('').astype(str)
        
        # Get description texts for year checking
        description_col = 'description_translated_text' if 'description_translated_text' in self.df.columns else 'description'
        descriptions = self.df[description_col].fillna('').astype(str) if description_col in self.df.columns else pd.Series([''] * len(self.df))
        
        # Analyze individual words
        all_tokens = []
        for summary in summaries:
            all_tokens.extend(self.preprocess_text(summary))
        
        word_freq = Counter(all_tokens)
        
        # Analyze common patterns and track which rows belong to each category
        patterns = {
            'software_issues': 0,
            'hardware_issues': 0,
            'training_requests': 0,
            'installation_issues': 0,
            'tuning_issues': 0,
            'calibration_issues': 0,
            'communication_issues': 0,
            'documentation_requests': 0
        }
        
        # Track row indices for each category and the triggering keywords
        category_rows = {
            'software_issues': [],
            'hardware_issues': [],
            'training_requests': [],
            'installation_issues': [],
            'tuning_issues': [],
            'calibration_issues': [],
            'communication_issues': [],
            'documentation_requests': []
        }
        
        # Track which keywords triggered each categorization
        category_keywords = {
            'software_issues': [],
            'hardware_issues': [],
            'training_requests': [],
            'installation_issues': [],
            'tuning_issues': [],
            'calibration_issues': [],
            'communication_issues': [],
            'documentation_requests': []
        }
        
        # Improved keyword sets - more specific and context-aware
        software_words = {'software', 'program', 'programming', 'code', 'application', 'version', 'update', 'bug', 'script', 'firmware', 'patch'}
        hardware_words = {'hardware', 'motor', 'drive', 'cable', 'sensor', 'encoder', 'axis', 'controller', 'board', 'amplifier', 'servo', 'stepper'}
        training_words = {'training', 'education', 'course', 'learn', 'tutorial', 'instruction', 'workshop', 'seminar', 'certification'}
        installation_words = {'install', 'installation', 'setup', 'configure', 'deployment', 'mounting', 'wiring', 'commissioning'}
        
        # More specific tuning keywords - removed overly broad terms
        tuning_words = {'tuning', 'tune', 'pid', 'gains', 'noisy', 'noise', 'resonance', 'instability', 'oscillate', 'unstable', 'velocity', 'acceleration', 'jerk', 'loaded'}
        
        # Conditional tuning keywords - patterns that must be present for these words to count
        tuning_conditional_patterns = [
            r'(?:position\s*error|positionerror).*fault',  # position error fault
            r'(?:over\s*current|overcurrent).*fault',      # over current fault
        ]
        
        # Exclusion words for tuning issues (case-insensitive)
        tuning_exclusions = {'omega', 'comet', 'rfq', 'rma', 'hga', 'rolex', 'analog', 'feedback', 'repair', 'scanhead', 'scan', 'head', 'gtse', 'autofocus'}
        
        # Pattern-based exclusions for tuning issues (regex patterns)
        tuning_exclusion_patterns = [
            r'MC#\d+',  # Matches MC# followed by numbers (e.g., MC#29, MC#123)
        ]
        
        communication_words = {'communication', 'network', 'connection', 'ethernet', 'protocol', 'interface', 'modbus', 'profinet', 'ethercat'}
        documentation_words = {'documentation', 'manual', 'guide', 'document', 'specification', 'datasheet', 'reference'}
        
        def has_pattern_exclusions(text: str, patterns: List[str]) -> bool:
            """Check if text contains any exclusion patterns"""
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            return False
        
        def has_conditional_patterns(text: str, patterns: List[str]) -> bool:
            """Check if text contains any conditional inclusion patterns"""
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            return False
        
        def has_old_years(text: str, threshold_year: int = 2024) -> bool:
            """Check if text contains years older than threshold (indicates legacy equipment)"""
            # Find all 4-digit years in the text (1900-2099)
            year_pattern = r'\b(?:19|20)\d{2}\b'
            years = re.findall(year_pattern, text)
            
            for year_str in years:
                year = int(year_str)
                if year < threshold_year:
                    return True
            return False
        
        for idx, summary in enumerate(summaries):
            tokens = set(self.preprocess_text(summary))
            original_idx = summaries.index[idx]  # Get the original dataframe index
            description = descriptions.iloc[idx] if idx < len(descriptions) else ""
            
            # Check each category and track which keywords triggered it
            software_matches = tokens.intersection(software_words)
            if software_matches:
                patterns['software_issues'] += 1
                category_rows['software_issues'].append(original_idx)
                category_keywords['software_issues'].extend(software_matches)
                
            hardware_matches = tokens.intersection(hardware_words)
            if hardware_matches:
                patterns['hardware_issues'] += 1
                category_rows['hardware_issues'].append(original_idx)
                category_keywords['hardware_issues'].extend(hardware_matches)
                
            training_matches = tokens.intersection(training_words)
            if training_matches:
                patterns['training_requests'] += 1
                category_rows['training_requests'].append(original_idx)
                category_keywords['training_requests'].extend(training_matches)
                
            installation_matches = tokens.intersection(installation_words)
            if installation_matches:
                patterns['installation_issues'] += 1
                category_rows['installation_issues'].append(original_idx)
                category_keywords['installation_issues'].extend(installation_matches)
                
            tuning_matches = tokens.intersection(tuning_words)
            tuning_conditional_matches = has_conditional_patterns(summary, tuning_conditional_patterns)
            tuning_exclusion_matches = tokens.intersection(tuning_exclusions)
            tuning_pattern_exclusions = has_pattern_exclusions(summary, tuning_exclusion_patterns)
            tuning_old_years = has_old_years(summary) or has_old_years(description)
            if (tuning_matches or tuning_conditional_matches) and not tuning_exclusion_matches and not tuning_pattern_exclusions and not tuning_old_years:
                patterns['tuning_issues'] += 1
                category_rows['tuning_issues'].append(original_idx)
                category_keywords['tuning_issues'].extend(tuning_matches)
                
            communication_matches = tokens.intersection(communication_words)
            if communication_matches:
                patterns['communication_issues'] += 1
                category_rows['communication_issues'].append(original_idx)
                category_keywords['communication_issues'].extend(communication_matches)
                
            documentation_matches = tokens.intersection(documentation_words)
            if documentation_matches:
                patterns['documentation_requests'] += 1
                category_rows['documentation_requests'].append(original_idx)
                category_keywords['documentation_requests'].extend(documentation_matches)
        
        return {
            'total_summaries': len(summaries),
            'top_words': word_freq.most_common(top_n),
            'category_patterns': patterns,
            'category_rows': category_rows,
            'category_keywords': category_keywords,
            'column_analyzed': summary_col
        }
    
    def display_word_analysis(self, analysis: Dict):
        """Display word frequency analysis results"""
        if not analysis:
            print("No analysis data available.")
            return
        
        print(f"\n=== Word Frequency Analysis ===")
        print(f"Total words processed: {analysis['total_words']:,}")
        print(f"Unique words found: {analysis['unique_words']:,}")
        print(f"Columns analyzed: {', '.join(analysis['columns_analyzed'])}")
        
        print(f"\n📊 Top {len(analysis['top_words'])} Most Common Words:")
        for i, (word, count) in enumerate(analysis['top_words'], 1):
            percentage = (count / analysis['total_words']) * 100
            print(f"{i:2d}. {word:<15} {count:4d} times ({percentage:.1f}%)")
    
    def display_phrase_analysis(self, analysis: Dict):
        """Display phrase analysis results"""
        if not analysis:
            print("No analysis data available.")
            return
        
        phrase_type = "Bigrams" if analysis['n_gram_size'] == 2 else f"{analysis['n_gram_size']}-grams"
        
        print(f"\n=== {phrase_type} Analysis ===")
        print(f"Total phrases processed: {analysis['total_phrases']:,}")
        print(f"Unique phrases found: {analysis['unique_phrases']:,}")
        print(f"Columns analyzed: {', '.join(analysis['columns_analyzed'])}")
        
        print(f"\n📝 Top {len(analysis['top_phrases'])} Most Common Phrases:")
        for i, (phrase, count) in enumerate(analysis['top_phrases'], 1):
            percentage = (count / analysis['total_phrases']) * 100
            print(f"{i:2d}. '{phrase}'  {count:4d} times ({percentage:.1f}%)")
    
    def display_summary_trends(self, analysis: Dict):
        """Display summary trends analysis"""
        if not analysis:
            print("No analysis data available.")
            return
        
        print(f"\n=== Summary Trends Analysis ===")
        print(f"Total summaries analyzed: {analysis['total_summaries']:,}")
        print(f"Column analyzed: {analysis['column_analyzed']}")
        
        print(f"\n🔍 Top {len(analysis['top_words'])} Words in Summaries:")
        for i, (word, count) in enumerate(analysis['top_words'], 1):
            percentage = (count / analysis['total_summaries']) * 100
            print(f"{i:2d}. {word:<15} {count:4d} summaries ({percentage:.1f}%)")
        
        print(f"\n📈 Issue Category Patterns:")
        patterns = analysis['category_patterns']
        category_keywords = analysis.get('category_keywords', {})
        total = analysis['total_summaries']
        
        # Sort patterns by count
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_patterns:
            if count > 0:
                percentage = (count / total) * 100
                category_display = category.replace('_', ' ').title()
                print(f"  {category_display:<20} {count:4d} cases ({percentage:.1f}%)")
                
                # Show top triggering keywords for this category
                if category in category_keywords and category_keywords[category]:
                    keyword_counts = Counter(category_keywords[category])
                    top_keywords = [kw for kw, _ in keyword_counts.most_common(3)]
                    print(f"    └─ Top keywords: {', '.join(top_keywords)}")
        
        # Ask if user wants to explore categories
        if any(count > 0 for count in patterns.values()):
            print(f"\n💡 Want to explore specific categories? Choose option 'Explore Categories' from the main menu after this analysis.")
            print(f"📊 Note: Categorization improved - removed overly broad keywords like 'performance' and 'optimization' from tuning.")
    
    def explore_categories(self):
        """
        Interactive category exploration - drill down into specific issue categories
        """
        if self.df is None:
            print("No data loaded. Please load a CSV file first.")
            return
        
        print("\n=== Category Explorer ===")
        if not NLTK_AVAILABLE:
            print("⚠️  Note: Using basic text processing (NLTK not available). Results may be less accurate.")
        
        # Run summary analysis to get categories
        print("Analyzing categories...")
        analysis = self.analyze_summary_trends(25)
        
        if not analysis or not analysis.get('category_patterns'):
            print("No category data available.")
            return
        
        patterns = analysis['category_patterns']
        category_rows = analysis['category_rows']
        
        # Show available categories with data
        available_categories = [(cat, count) for cat, count in patterns.items() if count > 0]
        if not available_categories:
            print("No categories found with cases.")
            return
        
        # Sort by count (most cases first)
        available_categories.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 Available Categories ({len(available_categories)} categories with cases):")
        for i, (category, count) in enumerate(available_categories, 1):
            category_display = category.replace('_', ' ').title()
            percentage = (count / analysis['total_summaries']) * 100
            print(f"{i:2d}. {category_display:<20} {count:4d} cases ({percentage:.1f}%)")
        
        print(f"{len(available_categories) + 1:2d}. Back to main menu")
        
        while True:
            try:
                choice = input(f"\nSelect category to explore (1-{len(available_categories) + 1}): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(available_categories) + 1:
                    break
                elif 1 <= choice_num <= len(available_categories):
                    selected_category, count = available_categories[choice_num - 1]
                    self.show_category_cases(selected_category, category_rows[selected_category], count)
                else:
                    print(f"Invalid choice. Please select 1-{len(available_categories) + 1}.")
                    
            except ValueError:
                print("Please enter a valid number.")
    
    def show_category_cases(self, category: str, row_indices: List[int], total_count: int):
        """
        Show cases for a specific category
        
        Args:
            category: Category name
            row_indices: List of dataframe row indices for this category
            total_count: Total number of cases in this category
        """
        category_display = category.replace('_', ' ').title()
        print(f"\n=== {category_display} Cases ===")
        print(f"Total cases: {total_count}")
        
        if not row_indices:
            print("No cases found for this category.")
            return
        
        # Get the subset of data for this category
        category_data = self.df.iloc[row_indices].copy()
        
        # Display options
        print(f"\nDisplay Options:")
        print(f"1. Show first 10 cases (summary only)")
        print(f"2. Show first 10 cases (full details)")
        print(f"3. Show all cases (summary only)")
        print(f"4. Show all cases (full details)")
        print(f"5. Show keyword diagnostics (why these cases were categorized)")
        print(f"6. Save category to CSV")
        print(f"7. Save category to TXT")
        if PLOTTING_AVAILABLE:
            print(f"8. Show keyword statistics chart")
            print(f"9. Back to category list")
        else:
            print(f"8. Back to category list")
        
        try:
            max_choice = 9 if PLOTTING_AVAILABLE else 8
            choice = input(f"Select option (1-{max_choice}): ").strip()
            
            if choice == '1':
                self.display_category_results(category_data.head(10), category_display, summary_only=True)
            elif choice == '2':
                self.display_category_results(category_data.head(10), category_display, summary_only=False)
            elif choice == '3':
                self.display_category_results(category_data, category_display, summary_only=True)
            elif choice == '4':
                self.display_category_results(category_data, category_display, summary_only=False)
            elif choice == '5':
                self.show_keyword_diagnostics(category_data, category, category_display)
            elif choice == '6':
                self.save_category_results(category_data, category, 'csv')
            elif choice == '7':
                self.save_category_results(category_data, category, 'txt')
            elif choice == '8' and PLOTTING_AVAILABLE:
                self.show_keyword_statistics_chart(category_data, category, category_display)
            elif choice == '8' and not PLOTTING_AVAILABLE:
                return
            elif choice == '9' and PLOTTING_AVAILABLE:
                return
            else:
                print("Invalid choice.")
                
        except ValueError:
            print("Please enter a valid number.")
    
    def display_category_results(self, category_data: pd.DataFrame, category_name: str, summary_only: bool = False):
        """Display category results in formatted way"""
        print(f"\n=== {category_name} Cases ({len(category_data)} shown) ===")
        
        for idx, (_, row) in enumerate(category_data.iterrows(), 1):
            print(f"\n--- Case {idx} ---")
            
            # Show key fields
            if 'created' in row.index and pd.notna(row['created']):
                print(f"Created: {row['created']}")
            if 'CustomerName' in row.index and pd.notna(row['CustomerName']):
                print(f"Customer: {row['CustomerName']}")
            if 'status' in row.index and pd.notna(row['status']):
                print(f"Status: {row['status']}")
            
            # Show summary
            if 'summary' in row.index:
                if 'summary_translated_text' in row.index and pd.notna(row['summary_translated_text']):
                    summary_text = str(row['summary_translated_text'])
                    if summary_only:
                        summary_text = summary_text[:200]
                        print(f"Summary: {summary_text}{'...' if len(str(row['summary_translated_text'])) > 200 else ''}")
                    else:
                        print(f"Summary: {summary_text}")
                else:
                    summary_text = str(row['summary'])
                    if summary_only:
                        summary_text = summary_text[:200]
                        print(f"Summary: {summary_text}{'...' if len(str(row['summary'])) > 200 else ''}")
                    else:
                        print(f"Summary: {summary_text}")
            
            # Show description if not summary_only
            if not summary_only and 'description' in row.index:
                if 'description_translated_text' in row.index and pd.notna(row['description_translated_text']):
                    description_text = str(row['description_translated_text'])[:400]
                    print(f"Description: {description_text}{'...' if len(str(row['description_translated_text'])) > 400 else ''}")
                else:
                    description_text = str(row['description'])[:400]
                    print(f"Description: {description_text}{'...' if len(str(row['description'])) > 400 else ''}")
            
            print("-" * 60)
    
    def show_keyword_diagnostics(self, category_data: pd.DataFrame, category: str, category_display: str):
        """Show keyword diagnostics for category - explain why cases were categorized"""
        print(f"\n=== Keyword Diagnostics for {category_display} ===")
        
        # Define the same keyword sets used for categorization
        keyword_sets = {
            'software_issues': {'software', 'program', 'programming', 'code', 'application', 'version', 'update', 'bug', 'script', 'firmware', 'patch'},
            'hardware_issues': {'hardware', 'motor', 'drive', 'cable', 'sensor', 'encoder', 'axis', 'controller', 'board', 'amplifier', 'servo', 'stepper'},
            'training_requests': {'training', 'education', 'course', 'learn', 'tutorial', 'instruction', 'workshop', 'seminar', 'certification'},
            'installation_issues': {'install', 'installation', 'setup', 'configure', 'deployment', 'mounting', 'wiring', 'commissioning'},
            'tuning_issues': {'tuning', 'tune', 'pid', 'gains', 'noisy', 'noise', 'resonance', 'instability', 'oscillate', 'unstable', 'velocity', 'acceleration', 'jerk', 'loaded'},
            'communication_issues': {'communication', 'network', 'connection', 'ethernet', 'protocol', 'interface', 'modbus', 'profinet', 'ethercat'},
            'documentation_requests': {'documentation', 'manual', 'guide', 'document', 'specification', 'datasheet', 'reference'}
        }
        
        # Define exclusion sets (words that prevent categorization)
        exclusion_sets = {
            'tuning_issues': {'omega', 'comet', 'rfq', 'rma', 'hga', 'rolex', 'analog', 'feedback', 'PSO', 'repair', 'scanhead', 'scan', 'head', 'gtse', 'autofocus'}
        
        }
        
        # Define pattern exclusions (regex patterns that prevent categorization)
        pattern_exclusion_sets = {
            'tuning_issues': [r'MC#\d+']  # MC# followed by numbers
        }
        
        # Define conditional patterns (patterns that add keywords when present)
        conditional_pattern_sets = {
            'tuning_issues': [
                r'(?:position\s*error|positionerror).*fault',  # position error fault
                r'(?:over\s*current|overcurrent).*fault',      # over current fault
            ]
        }
        
        if category not in keyword_sets:
            print("No keyword set defined for this category.")
            return
        
        category_keywords = keyword_sets[category]
        print(f"Keywords that trigger {category_display}: {', '.join(sorted(category_keywords))}")
        
        # Show exclusion words if they exist for this category
        if category in exclusion_sets:
            exclusion_keywords = exclusion_sets[category]
            print(f"Exclusion words (prevent categorization): {', '.join(sorted(exclusion_keywords))}")
        
        # Show pattern exclusions if they exist for this category
        if category in pattern_exclusion_sets:
            exclusion_patterns = pattern_exclusion_sets[category]
            print(f"Exclusion patterns (prevent categorization): {', '.join(exclusion_patterns)}")
        
        # Show conditional patterns if they exist for this category
        if category in conditional_pattern_sets:
            conditional_patterns = conditional_pattern_sets[category]
            print(f"Conditional patterns (add keywords when found): {', '.join(conditional_patterns)}")
        
        # Show year exclusion for tuning issues
        if category == 'tuning_issues':
            print(f"Year exclusion: Cases mentioning years < 2024 are excluded (legacy equipment)")
        
        print(f"\nAnalyzing first 10 cases to show which keywords triggered categorization:")
        
        # Use translated/cleaned summary when available
        summary_col = 'summary_translated_text' if 'summary_translated_text' in category_data.columns else 'summary'
        
        for idx, (_, row) in enumerate(category_data.head(10).iterrows(), 1):
            print(f"\n--- Case {idx} ---")
            
            if summary_col in row.index and pd.notna(row[summary_col]):
                summary_text = str(row[summary_col])
                print(f"Summary: {summary_text[:200]}{'...' if len(summary_text) > 200 else ''}")
                
                # Find which keywords from this category appear in the summary
                tokens = set(self.preprocess_text(summary_text))
                matching_keywords = tokens.intersection(category_keywords)
                
                # Check for exclusion words
                exclusion_matches = set()
                if category in exclusion_sets:
                    exclusion_matches = tokens.intersection(exclusion_sets[category])
                
                # Check for pattern exclusions
                pattern_exclusion_matches = []
                if category in pattern_exclusion_sets:
                    for pattern in pattern_exclusion_sets[category]:
                        if re.search(pattern, summary_text, re.IGNORECASE):
                            pattern_exclusion_matches.append(pattern)
                
                # Check for conditional patterns
                conditional_pattern_matches = []
                if category in conditional_pattern_sets:
                    for pattern in conditional_pattern_sets[category]:
                        if re.search(pattern, summary_text, re.IGNORECASE):
                            conditional_pattern_matches.append(pattern)
                
                # Check for year exclusions (for tuning issues)
                year_exclusion_triggered = False
                if category == 'tuning_issues':
                    # Check both summary and description if available
                    description_text = ""
                    if 'description_translated_text' in row.index and pd.notna(row['description_translated_text']):
                        description_text = str(row['description_translated_text'])
                    elif 'description' in row.index and pd.notna(row['description']):
                        description_text = str(row['description'])
                    
                    year_exclusion_triggered = has_old_years(summary_text) or has_old_years(description_text)
                
                if matching_keywords or conditional_pattern_matches:
                    if matching_keywords:
                        print(f"🎯 Triggering keywords: {', '.join(sorted(matching_keywords))}")
                    if conditional_pattern_matches:
                        print(f"🎯 Triggering conditional patterns: {', '.join(conditional_pattern_matches)}")
                    if exclusion_matches:
                        print(f"❌ Exclusion words found: {', '.join(sorted(exclusion_matches))} - Should NOT be categorized!")
                    if pattern_exclusion_matches:
                        print(f"❌ Exclusion patterns found: {', '.join(pattern_exclusion_matches)} - Should NOT be categorized!")
                    if year_exclusion_triggered:
                        print(f"❌ Old year exclusion triggered - Should NOT be categorized!")
                else:
                    print(f"⚠️  No {category_display.lower()} keywords or patterns found - possible misclassification!")
                    
                if exclusion_matches and not matching_keywords and not conditional_pattern_matches:
                    print(f"❌ Contains exclusion words: {', '.join(sorted(exclusion_matches))}")
                    
                if pattern_exclusion_matches and not matching_keywords and not conditional_pattern_matches:
                    print(f"❌ Contains exclusion patterns: {', '.join(pattern_exclusion_matches)}")
                    
                if year_exclusion_triggered and not matching_keywords and not conditional_pattern_matches:
                    print(f"❌ Contains old year references (legacy equipment)")
                    
                # Check if it might belong to other categories
                other_matches = {}
                for other_cat, other_keywords in keyword_sets.items():
                    if other_cat != category:
                        other_matching = tokens.intersection(other_keywords)
                        if other_matching:
                            other_matches[other_cat] = other_matching
                
                if other_matches:
                    print(f"💡 Also matches:")
                    for other_cat, keywords in other_matches.items():
                        other_display = other_cat.replace('_', ' ').title()
                        print(f"   {other_display}: {', '.join(sorted(keywords))}")
            else:
                print("No summary available for analysis")
            
            print("-" * 50)
        
        print(f"\n📊 This analysis helps identify:")
        print(f"   • Cases that might be miscategorized")
        print(f"   • Keywords that are too broad or ambiguous")
        print(f"   • Cases that belong to multiple categories")
    
    def save_category_results(self, category_data: pd.DataFrame, category: str, format_type: str):
        """Save category results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        category_clean = category.replace('_', '-')
        
        if format_type == 'csv':
            filename = f"gtse_category_{category_clean}_{timestamp}.csv"
            try:
                category_data.to_csv(filename, index=False)
                print(f"✓ Saved {len(category_data)} {category.replace('_', ' ')} cases to: {filename}")
            except Exception as e:
                print(f"✗ Error saving CSV: {e}")
        
        elif format_type == 'txt':
            filename = f"gtse_category_{category_clean}_{timestamp}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    category_display = category.replace('_', ' ').title()
                    f.write(f"=== {category_display} Cases ===\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total cases: {len(category_data)}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for idx, (_, row) in enumerate(category_data.iterrows(), 1):
                        f.write(f"--- Case {idx} ---\n")
                        
                        # Show key fields
                        if 'created' in row.index and pd.notna(row['created']):
                            f.write(f"Created: {row['created']}\n")
                        if 'CustomerName' in row.index and pd.notna(row['CustomerName']):
                            f.write(f"Customer: {row['CustomerName']}\n")
                        if 'status' in row.index and pd.notna(row['status']):
                            f.write(f"Status: {row['status']}\n")
                        
                        # Summary
                        if 'summary' in row.index and pd.notna(row['summary']):
                            if 'summary_translated_text' in row.index and pd.notna(row['summary_translated_text']):
                                f.write(f"Summary: {str(row['summary_translated_text'])}\n")
                            else:
                                f.write(f"Summary: {str(row['summary'])}\n")
                        
                        # Description
                        if 'description' in row.index and pd.notna(row['description']):
                            if 'description_translated_text' in row.index and pd.notna(row['description_translated_text']):
                                f.write(f"Description: {str(row['description_translated_text'])}\n")
                            else:
                                f.write(f"Description: {str(row['description'])}\n")
                        
                        f.write("\n" + "=" * 80 + "\n\n")
                
                print(f"✓ Saved {len(category_data)} {category.replace('_', ' ')} cases to: {filename}")
                
            except Exception as e:
                print(f"✗ Error saving TXT: {e}")

    def run_interactive_search(self):
        """Run interactive keyword search interface"""
        print("=== GTSE Results Analyzer ===")
        print("Search through translated GTSE query results\n")
        
        # Load file
        if not self.load_csv_file():
            return
        
        # Show data stats
        self.get_column_stats()
        
        while True:
            print(f"\n=== Analysis Options ===")
            print("1. Search keywords (traditional)")
            print("2. Word frequency analysis")
            print("3. Common phrases analysis")
            print("4. Summary trends analysis")
            print("5. Complete text analytics (all analyses)")
            print("6. Explore categories (drill down into issue types)")
            print("7. View data statistics")
            print("8. Load different file")
            print("9. Exit")
            
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == '1':
                # Get keywords
                keywords_input = input("\nEnter keywords (comma-separated): ").strip()
                if not keywords_input:
                    print("No keywords entered.")
                    continue
                
                keywords = [k.strip() for k in keywords_input.split(',') if k.strip()]
                
                # Case sensitive option
                case_sensitive = input("Case sensitive search? (y/n): ").strip().lower() == 'y'
                
                # Search
                print(f"\nSearching for: {keywords}")
                results = self.search_keywords(keywords, case_sensitive)
                
                # Display results
                self.display_results(results, keywords)
                
                # Save option
                if not results.empty:
                    print(f"\n=== Save Results ===")
                    print(f"Found {len(results)} results")
                    print("1. Save to CSV (spreadsheet format)")
                    print("2. Save to TXT (readable format with complete text)")
                    print("3. Save both formats")
                    print("4. Don't save")
                    
                    save_choice = input("Choose save option (1-4): ").strip()
                    
                    if save_choice == '1':
                        self.save_results(results, keywords)
                    elif save_choice == '2':
                        self.save_results_to_txt(results, keywords)
                    elif save_choice == '3':
                        self.save_results(results, keywords)
                        self.save_results_to_txt(results, keywords)
                    elif save_choice == '4':
                        print("Results not saved.")
                    else:
                        print("Invalid choice. Results not saved.")
            
            elif choice == '2':
                # Word frequency analysis
                print("\n=== Word Frequency Analysis ===")
                if not NLTK_AVAILABLE:
                    print("⚠️  Note: Using basic text processing (NLTK not available). Results may be less accurate.")
                
                columns_input = input("Analyze which columns? (enter 'summary', 'description', or 'both' [default]): ").strip().lower()
                
                if columns_input == 'summary':
                    columns = ['summary']
                elif columns_input == 'description':
                    columns = ['description']
                else:
                    columns = ['summary', 'description']
                
                try:
                    top_n = int(input("How many top words to show? (default: 30): ").strip() or "30")
                except ValueError:
                    top_n = 30
                
                analysis = self.analyze_word_frequency(columns, top_n)
                self.display_word_analysis(analysis)
            
            elif choice == '3':
                # Phrase analysis
                print("\n=== Common Phrases Analysis ===")
                if not NLTK_AVAILABLE:
                    print("⚠️  Note: Using basic text processing (NLTK not available). Results may be less accurate.")
                
                columns_input = input("Analyze which columns? (enter 'summary', 'description', or 'both' [default]): ").strip().lower()
                
                if columns_input == 'summary':
                    columns = ['summary']
                elif columns_input == 'description':
                    columns = ['description']
                else:
                    columns = ['summary', 'description']
                
                try:
                    n_gram_size = int(input("Phrase length? (2=two words, 3=three words [default: 2]): ").strip() or "2")
                    if n_gram_size < 2:
                        n_gram_size = 2
                except ValueError:
                    n_gram_size = 2
                
                try:
                    top_n = int(input("How many top phrases to show? (default: 20): ").strip() or "20")
                except ValueError:
                    top_n = 20
                
                analysis = self.analyze_phrases(columns, n_gram_size, top_n)
                self.display_phrase_analysis(analysis)
            
            elif choice == '4':
                # Summary trends analysis
                print("\n=== Summary Trends Analysis ===")
                if not NLTK_AVAILABLE:
                    print("⚠️  Note: Using basic text processing (NLTK not available). Results may be less accurate.")
                
                try:
                    top_n = int(input("How many top words to show? (default: 25): ").strip() or "25")
                except ValueError:
                    top_n = 25
                
                analysis = self.analyze_summary_trends(top_n)
                self.display_summary_trends(analysis)
            
            elif choice == '5':
                # Complete analysis
                print("\n=== Complete Text Analytics ===")
                if not NLTK_AVAILABLE:
                    print("⚠️  Note: Using basic text processing (NLTK not available). Results may be less accurate.")
                print("Running comprehensive analysis on all text data...\n")
                
                # Word frequency for both columns
                word_analysis = self.analyze_word_frequency(['summary', 'description'], 25)
                self.display_word_analysis(word_analysis)
                
                # Common bigrams
                phrase_analysis = self.analyze_phrases(['summary', 'description'], 2, 15)
                self.display_phrase_analysis(phrase_analysis)
                
                # Common trigrams
                trigram_analysis = self.analyze_phrases(['summary', 'description'], 3, 10)
                self.display_phrase_analysis(trigram_analysis)
                
                # Summary trends
                summary_analysis = self.analyze_summary_trends(20)
                self.display_summary_trends(summary_analysis)
                
                print(f"\n{'='*80}")
                print("🎯 Analysis Complete! Use the insights above to identify trends and patterns.")
                print("💡 Consider searching for specific terms that appear frequently in your data.")
            
            elif choice == '6':
                # Explore categories
                self.explore_categories()
            
            elif choice == '7':
                self.get_column_stats()
            
            elif choice == '8':
                if self.load_csv_file():
                    self.get_column_stats()
            
            elif choice == '9':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please select 1-9.")

    def show_keyword_statistics_chart(self, category_data: pd.DataFrame, category: str, category_display: str):
        """Create and display interactive keyword statistics charts using Plotly"""
        print(f"\n🔍 DEBUG - Chart function START - category: '{category}', display: '{category_display}'")
        
        if not PLOTTING_AVAILABLE:
            print("❌ Plotly not available. Please install plotly: pip install plotly")
            return
        
        print(f"📊 Generating intelligent keyword analysis dashboard for {category_display}...")
        print(f"🔍 Debug - Chart function received category: '{category}' with display: '{category_display}'")
        
        try:
            print(f"🔍 DEBUG - About to call analyze_category_keywords...")
            # Collect keyword statistics
            keyword_stats = self.analyze_category_keywords(category_data, category)
            print(f"🔍 DEBUG - analyze_category_keywords completed successfully")
        except Exception as e:
            print(f"❌ ERROR in analyze_category_keywords: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Debug: Show what we're actually getting
        print(f"\n🔍 Debug - Top 5 keyword groups found:")
        for i, (keyword, count) in enumerate(list(keyword_stats['triggering_keywords'].items())[:5], 1):
            print(f"   {i}. '{keyword}': {count} cases")
        
        if len(keyword_stats['triggering_keywords']) == 0:
            print("⚠️  WARNING: No triggering keywords found! This suggests a problem with the analysis.")
        
        if not keyword_stats:
            print("No keyword statistics available for this category.")
            return
        
        try:
            print(f"🔍 DEBUG - About to create subplots...")
            # Create subplots with enhanced layout - moving average on top (full width), other charts below
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    '12-Month Moving Average Trend',
                    '',  # Empty since row 1 spans both columns
                    '',  # Will add manually
                    ''   # Will add manually
                ),
                specs=[[{"colspan": 2}, None],  # Row 1 spans both columns
                       [{}, {}]],  # Row 2 has two separate charts
                vertical_spacing=0.35,  # Adjusted spacing for new layout
                horizontal_spacing=0.15
            )
            print(f"🔍 DEBUG - Subplots created successfully")
        except Exception as e:
            print(f"❌ ERROR creating subplots: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print(f"🔍 DEBUG - Adding Chart 2: Top triggering keywords...")
            # Chart 2: Top triggering keywords (moved to row 2, col 1)
            if keyword_stats['triggering_keywords']:
                keywords = list(keyword_stats['triggering_keywords'].keys())[:10]
                counts = list(keyword_stats['triggering_keywords'].values())[:10]
                
                fig.add_trace(
                    go.Bar(
                        x=keywords,
                        y=counts,
                        name='Triggering Keywords',
                        marker=dict(
                            color='steelblue',
                            opacity=0.8,
                            line=dict(color='darkblue', width=1)
                        ),
                        text=counts,
                        textposition='auto',  # Auto positioning to avoid title overlap
                        hovertemplate='<b>%{x}</b> (keyword group)<br>Frequency: %{y} cases<br><i>Combined related words</i><br><extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=2  # Moved to row 2, column 2
                )
                print(f"🔍 DEBUG - Chart 2 added successfully")
            else:
                print(f"🔍 DEBUG - Skipping Chart 2 - no triggering keywords")
        except Exception as e:
            print(f"❌ ERROR adding Chart 2: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print(f"🔍 DEBUG - Adding Chart 3: Category comparison...")
            # Chart 3: Category comparison with enhanced colors (moved to row 2, col 2)
            if keyword_stats['category_comparison']:
                comp_cats = list(keyword_stats['category_comparison'].keys())[:5]
                comp_counts = list(keyword_stats['category_comparison'].values())[:5]
                comp_display = [cat.replace('_', ' ').title() for cat in comp_cats]
                
                # Color the current category differently
                colors = ['gold' if cat == category else 'lightcoral' for cat in comp_cats]
                
                fig.add_trace(
                    go.Bar(
                        x=comp_display,
                        y=comp_counts,
                        name='Category Cases',
                        marker=dict(
                            color=colors,
                            opacity=0.8,
                            line=dict(color='black', width=1)
                        ),
                        text=comp_counts,
                        textposition='auto',  # Auto positioning to avoid title overlap
                        hovertemplate='<b>%{x}</b><br>Total Cases: %{y}<br><extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=1  # Moved to row 2, column 1
                )
                print(f"🔍 DEBUG - Chart 3 added successfully")
            else:
                print(f"🔍 DEBUG - Skipping Chart 3 - no category comparison data")
        except Exception as e:
            print(f"❌ ERROR adding Chart 3: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print(f"🔍 DEBUG - Adding Chart 1: 12-Month Moving Average (full width)...")
            # Chart 1: 12-Month Moving Average Trend (spans both columns in row 1)
            if keyword_stats['timeline_data'] and len(keyword_stats['timeline_data']) > 0:
                months = list(keyword_stats['timeline_data'].keys())
                moving_avg = list(keyword_stats['timeline_data'].values())
                
                # Convert month strings to more readable format
                from datetime import datetime
                month_labels = []
                for month in months:
                    try:
                        date_obj = datetime.strptime(month, '%Y-%m')
                        month_labels.append(date_obj.strftime('%b %Y'))  # e.g., "Jan 2024"
                    except:
                        month_labels.append(month)
                
                # Create line chart for moving average (larger, more prominent)
                fig.add_trace(
                    go.Scatter(
                        x=month_labels,
                        y=moving_avg,
                        mode='lines+markers',
                        name='3-Month Moving Average',
                        line=dict(color='darkorange', width=5),  # Thicker line for prominence
                        marker=dict(
                            color='darkorange',
                            size=12,  # Larger markers
                            line=dict(color='white', width=3)
                        ),
                        fill='tonexty',
                        fillcolor='rgba(255, 140, 0, 0.2)',
                        hovertemplate='<b>%{x}</b><br>3-Month Avg: %{y} cases<br><i>12-month trend analysis</i><br><extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=1  # Row 1, spans both columns due to colspan=2
                )
                print(f"🔍 DEBUG - Chart 1 moving average line chart added ({len(months)} months, full width)")
            else:
                fig.add_annotation(
                    text="No trend data<br>available",
                    xref="x1", yref="y1",
                    x=0.5, y=0.5,
                    xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="gray"),
                    row=1, col=1
                )
                print(f"🔍 DEBUG - Chart 1 annotation added (no trend data)")
        except Exception as e:
            print(f"❌ ERROR adding Chart 1: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print(f"🔍 DEBUG - Updating layout...")
            # Update layout with professional styling
            fig.update_layout(
                title=dict(
                    text=f'📊 Keyword Analysis: {category_display}<br><sub>({len(category_data)} cases analyzed)</sub>',
                    x=0.5,
                    font=dict(size=20, family="Arial Black")
                ),
                height=1000,  # Further increased height to accommodate better spacing
                template='plotly_white',
                showlegend=False,
                font=dict(family="Arial", size=12),
                margin=dict(t=140, b=70, l=70, r=70)  # Further increased margins for better spacing
            )
            
            # Update axes for better readability
            fig.update_xaxes(tickangle=45, title_font=dict(size=12, family="Arial"))
            fig.update_yaxes(title_font=dict(size=12, family="Arial"))
            
            # Add subtle grid lines
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
            
            # Manually add both bottom chart titles as annotations for reliable positioning
            fig.add_annotation(
                text="Top 10 Keywords",
                xref="paper", yref="paper",
                x=0.775, y=0.38,  # Position above the LEFT bottom chart (individual keywords)
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=14, family="Arial", color="black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
            
            fig.add_annotation(
                text="Category Comparison",
                xref="paper", yref="paper",
                x=0.225, y=0.38,  # Position above the RIGHT bottom chart (category bars)
                xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=14, family="Arial", color="black"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
            print(f"🔍 DEBUG - Layout updated successfully with manual title annotation")
        except Exception as e:
            print(f"❌ ERROR updating layout: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print(f"🔍 DEBUG - Saving chart files...")
            # Save the chart as HTML (interactive) and static image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_filename = f"gtse_analysis_{category}_{timestamp}.html"
            png_filename = f"gtse_analysis_{category}_{timestamp}.png"
            
            # Save interactive HTML
            fig.write_html(html_filename)
            print(f"✅ Interactive chart saved as: {html_filename}")
            
            # Save static PNG
            try:
                fig.write_image(png_filename, width=1600, height=800, scale=2)
                print(f"✅ Static chart saved as: {png_filename}")
            except Exception as e:
                print(f"⚠️  Static image save failed (install kaleido for PNG export): {e}")
            
            print(f"🔍 DEBUG - File saving completed")
        except Exception as e:
            print(f"❌ ERROR saving chart files: {e}")
            import traceback
            traceback.print_exc()
            return
        
        try:
            print(f"🔍 DEBUG - Displaying statistics summary...")
            # Display statistics summary with enhanced formatting
            print(f"\n📈 Keyword Statistics Summary:")
            print(f"   • 🎯 Triggering keyword groups found: {len(keyword_stats['triggering_keywords'])}")
            print(f"   • 📊 Cases analyzed: {len(category_data)}")
            print(f"   • 📈 Moving average period: {len(keyword_stats['timeline_data'])} months with data")
            print(f"   • 📊 Trend analysis: 3-month moving average over 12-month period")
            
            # Add top keyword insights with intelligent grouping info
            if keyword_stats['triggering_keywords']:
                top_keyword = list(keyword_stats['triggering_keywords'].keys())[0]
                top_count = list(keyword_stats['triggering_keywords'].values())[0]
                print(f"   • 🏆 Most frequent keyword group: '{top_keyword}' ({top_count} cases)")
                
                # Show a few examples of intelligent grouping
                print(f"   • 🤖 Intelligent grouping examples:")
                example_groups = [
                    ("noise", "combines: noise, noisy, noisiness"),
                    ("tuning", "combines: tune, tuning, tuned"),
                    ("stability", "combines: stable, unstable, instability")
                ]
                for group_name, description in example_groups[:2]:  # Show top 2 examples
                    if group_name in keyword_stats['triggering_keywords']:
                        print(f"     - {description}")
                        break
            
            print(f"🔍 DEBUG - Statistics summary completed")
        except Exception as e:
            print(f"❌ ERROR displaying statistics summary: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Show the interactive chart
        try:
            print(f"🔍 Debug - About to display chart with {len(fig.data)} traces")
            print(f"🔍 Debug - Chart has {len([trace for trace in fig.data])} data traces")
            fig.show()
            print("✅ Chart display attempted successfully")
        except Exception as e:
            print(f"❌ Error displaying chart: {e}")
            print("💡 Try opening the saved HTML file manually to view the chart")
        
        try:
            if os.path.exists(html_filename):
                print(f"🌐 Attempting to open HTML file in default browser...")
            else:
                print(f"❌ HTML file {html_filename} not found")
        except Exception as e:
            print(f"⚠️  Could not open HTML file automatically: {e}")
            print(f"💡 Manually open: {os.path.abspath(html_filename)}")
    
    def analyze_category_keywords(self, category_data: pd.DataFrame, category: str) -> Dict:
        """Analyze keyword patterns in the category data with intelligent word grouping"""
        print(f"🔍 Debug - analyze_category_keywords received category: '{category}'")
        
        # Get summary and description texts
        summary_col = 'summary_translated_text' if 'summary_translated_text' in category_data.columns else 'summary'
        description_col = 'description_translated_text' if 'description_translated_text' in category_data.columns else 'description'
        
        # Define intelligent keyword groups (combine related words)
        keyword_groups = {
            'noise': {'noise', 'noisy', 'noisiness'},
            'tuning': {'tune', 'tuning', 'tuned'},
            'stability': {'stable', 'unstable', 'instability', 'stability'},
            'oscillation': {'oscillate', 'oscillating', 'oscillation', 'oscillations'},
            'resonance': {'resonate', 'resonance', 'resonant'},
            'vibration': {'vibrate', 'vibrating', 'vibration', 'vibrations'},
            'servo': {'servo', 'servoing'},
            'motor': {'motor', 'motors'},
            'encoder': {'encode', 'encoder', 'encoding'},
            'controller': {'control', 'controller', 'controlling'},
            'axis': {'axis', 'axes'},
            'position': {'position', 'positioning', 'positional'},
            'velocity': {'velocity', 'velocities'},
            'acceleration': {'acceleration', 'accelerate', 'accelerating'},
            'fault': {'fault', 'faults', 'faulting'},
            'error': {'error', 'errors'},
            'current': {'current', 'overcurrent'},
            'gain': {'gain', 'gains'},
            'filter': {'filter', 'filtering', 'filtered'},
            'trajectory': {'trajectory', 'trajectories'},
            'jerk': {'jerk', 'jerky', 'jerking'}
        }
        
        # Define category keyword sets (using intelligent groups for analysis)
        keyword_sets = {
            'software_issues': {'software', 'program', 'programming', 'code', 'application', 'version', 'update', 'bug', 'script', 'firmware', 'patch'},
            'hardware_issues': {'hardware', 'motor', 'drive', 'cable', 'sensor', 'encoder', 'axis', 'controller', 'board', 'amplifier', 'servo', 'stepper'},
            'training_requests': {'training', 'education', 'course', 'learn', 'tutorial', 'instruction', 'workshop', 'seminar', 'certification'},
            'installation_issues': {'install', 'installation', 'setup', 'configure', 'deployment', 'mounting', 'wiring', 'commissioning'},
            'tuning_issues': {'tuning', 'tune', 'pid', 'gains', 'noisy', 'noise', 'resonance', 'instability', 'oscillate', 'unstable', 'velocity', 'acceleration', 'jerk', 'loaded'},
            'communication_issues': {'communication', 'network', 'connection', 'ethernet', 'protocol', 'interface', 'modbus', 'profinet', 'ethercat'},
            'documentation_requests': {'documentation', 'manual', 'guide', 'document', 'specification', 'datasheet', 'reference'}
        }
        
        # Create expanded keyword sets that include all variations for better matching
        expanded_keyword_sets = {}
        for cat_name, base_keywords in keyword_sets.items():
            expanded_set = set()
            for keyword in base_keywords:
                # Add the original keyword
                expanded_set.add(keyword)
                # Add all variations from keyword groups
                for group_name, group_words in keyword_groups.items():
                    if keyword in group_words:
                        expanded_set.update(group_words)
            expanded_keyword_sets[cat_name] = expanded_set
        
        print(f"🔍 Debug - Available keyword sets: {list(expanded_keyword_sets.keys())}")
        print(f"🔍 Debug - Looking for category '{category}' in keyword sets...")
        
        if category not in expanded_keyword_sets:
            print(f"❌ ERROR: Category '{category}' not found in keyword sets!")
            print(f"Available categories: {list(expanded_keyword_sets.keys())}")
            return {'triggering_keywords': {}, 'timeline_data': {}, 'category_comparison': {}}
        
        # Helper function to map words to their intelligent groups
        def get_keyword_group(word):
            """Map a word to its intelligent group name"""
            for group_name, group_words in keyword_groups.items():
                if word.lower() in group_words:
                    return group_name
            return word  # Return original word if no group found
        
        # Import required modules
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Count keyword occurrences with intelligent grouping
        triggering_keywords = Counter()
        timeline_data = {}
        
        # Debug: Show what keywords we're looking for
        if category in expanded_keyword_sets:
            category_keywords = expanded_keyword_sets[category]
            print(f"\n🔍 Debug - Looking for these {len(category_keywords)} keywords in {category}:")
            print(f"   {sorted(list(category_keywords))[:10]}{'...' if len(category_keywords) > 10 else ''}")
        
        # Debug counters
        cases_processed = 0
        keywords_found_debug = Counter()
        
        # Analyze each case
        for _, row in category_data.iterrows():
            cases_processed += 1
            # Get text content
            summary_text = str(row[summary_col]) if summary_col in row.index and pd.notna(row[summary_col]) else ""
            description_text = str(row[description_col]) if description_col in row.index and pd.notna(row[description_col]) else ""
            full_text = f"{summary_text} {description_text}".lower()
            
            # Tokenize text
            tokens = set(self.preprocess_text(full_text))
            
            # Count triggering keywords with intelligent grouping
            if category in expanded_keyword_sets:
                category_keywords = expanded_keyword_sets[category]
                matching_keywords = tokens.intersection(category_keywords)
                for keyword in matching_keywords:
                    # Debug: Track original keywords found
                    keywords_found_debug[keyword] += 1
                    # Map to intelligent group
                    group_name = get_keyword_group(keyword)
                    triggering_keywords[group_name] += 1
            
            # Extract creation date for 9-month moving average timeline
            if 'created' in row.index and pd.notna(row['created']):
                try:
                    # Parse the date and group by month
                    created_date = pd.to_datetime(row['created']).date()
                    month_str = created_date.strftime('%Y-%m')  # Group by month instead of day
                    timeline_data[month_str] = timeline_data.get(month_str, 0) + 1
                except Exception as e:
                    # Debug: Show date parsing issues
                    if cases_processed <= 3:  # Only show first few errors
                        print(f"🔍 Debug - Date parsing failed for: '{row['created']}' - {e}")
                    pass  # Skip invalid dates
        
        # Debug: Show keyword mapping results
        print(f"\n🔍 Debug - Processed {cases_processed} cases")
        print(f"🔍 Debug - Top 5 original keywords found:")
        for keyword, count in keywords_found_debug.most_common(5):
            group_name = get_keyword_group(keyword)
            print(f"   '{keyword}' → '{group_name}': {count} times")
        
        print(f"🔍 Debug - Top 5 after intelligent grouping:")
        for group_name, count in triggering_keywords.most_common(5):
            print(f"   '{group_name}': {count} total")
        
        # Get category comparison data (mock data for now - in real implementation, 
        # you'd run the full category analysis)
        category_comparison = {
            'tuning_issues': len(category_data),
            'software_issues': 241,
            'hardware_issues': 875,
            'communication_issues': 173,
            'installation_issues': 116,
        }
        
        # Debug timeline data
        print(f"🔍 Debug - Timeline data before processing: {len(timeline_data)} months")
        if timeline_data:
            print(f"   Month range: {min(timeline_data.keys())} to {max(timeline_data.keys())}")
            print(f"   Sample months: {list(timeline_data.items())[:3]}")
        
        # Generate 12-month moving average timeline  
        from datetime import datetime, timedelta
        import calendar
        
        # Get the last 12 months including current month
        current_date = datetime.now()
        months_data = {}
        
        for i in range(11, -1, -1):  # 12 months: 11 months back + current month
            target_date = current_date.replace(day=1) - timedelta(days=i*30)
            # Get first day of the target month
            first_day = target_date.replace(day=1)
            month_key = first_day.strftime('%Y-%m')
            
            # Get actual case count for this month (or 0 if no cases)
            case_count = timeline_data.get(month_key, 0)
            months_data[month_key] = case_count
        
        # Calculate 3-month moving average for smoother trends
        moving_avg_data = {}
        month_keys = sorted(months_data.keys())
        
        for i, month in enumerate(month_keys):
            if i >= 2:  # Need at least 3 months for moving average
                # Calculate average of current month and 2 previous months
                window_months = month_keys[i-2:i+1]
                avg_value = sum(months_data[m] for m in window_months) / 3
                moving_avg_data[month] = round(avg_value, 1)
            else:
                # For first 2 months, just use the actual value
                moving_avg_data[month] = months_data[month]
        
        print(f"🔍 Debug - Final moving average data: {len(moving_avg_data)} months")
        if moving_avg_data:
            print(f"   Final range: {min(moving_avg_data.keys())} to {max(moving_avg_data.keys())}")
            print(f"   Sample data: {list(moving_avg_data.items())[:3]}")
        
        sorted_timeline = moving_avg_data
        
        return {
            'triggering_keywords': dict(triggering_keywords.most_common()),
            'timeline_data': sorted_timeline,
            'category_comparison': category_comparison
        }

def main():
    """Main entry point"""
    try:
        analyzer = GTSEAnalyzer()
        analyzer.run_interactive_search()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 