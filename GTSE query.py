from google.cloud import bigquery
from deep_translator import GoogleTranslator
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import logging
import re
from typing import Dict, List, Optional
import time
from datetime import datetime, date
import os
import sys

# Import the generic BigQuery client
from bigquery_client import GenericBigQueryClient

class GTSEQueryTranslator:
    def __init__(self):
        print("1. Starting GTSEQueryTranslator init...")
        
        print("2. Creating BigQuery client...")
        self.bigquery_client = GenericBigQueryClient()
        print("3. BigQuery client created successfully")
        
        print("4. Creating Google Translator...")
        self.translator = GoogleTranslator(source='auto', target='en')
        print("5. Google Translator created successfully")
        
        print("6. Setting up logging...")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        print("7. GTSEQueryTranslator init complete")
    
    def clean_jira_text(self, text: str) -> str:
        """
        Clean Jira formatting markup from text
        
        Args:
            text: Raw text with Jira markup
            
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return text
        
        clean_text = str(text)
        
        # Remove Jira color formatting tags like {color:#000000} or {color:red}
        clean_text = re.sub(r'\{color[^}]*\}', '', clean_text)
        
        # Remove other Jira formatting tags like {panel}, {code}, {quote}, etc.
        clean_text = re.sub(r'\{[^}]*\}', '', clean_text)
        
        # Replace various forms of line endings with spaces
        clean_text = clean_text.replace('\\r\\n', ' ')  # Escaped version
        clean_text = clean_text.replace('\r\n', ' ')    # Actual line breaks
        clean_text = clean_text.replace('\\n', ' ')     # Just newlines
        clean_text = clean_text.replace('\n', ' ')      # Actual newlines
        clean_text = clean_text.replace('\\r', ' ')     # Just carriage returns
        clean_text = clean_text.replace('\r', ' ')      # Actual carriage returns
        
        # Remove table formatting pipes and other table markup
        clean_text = re.sub(r'\s*\|\s*', ' ', clean_text)  # Table cell separators
        clean_text = re.sub(r'\|\|', ' ', clean_text)      # Table headers
        
        # Remove Jira links and markup
        clean_text = re.sub(r'\[([^\]]*)\|([^\]]*)\]', r'\1', clean_text)  # [text|link] -> text
        clean_text = re.sub(r'\[([^\]]*)\]', r'\1', clean_text)            # [text] -> text
        
        # Remove common Jira markup patterns
        clean_text = re.sub(r'\*([^*]*)\*', r'\1', clean_text)  # *bold* -> bold
        clean_text = re.sub(r'_([^_]*)_', r'\1', clean_text)    # _italic_ -> italic
        clean_text = re.sub(r'\+([^+]*)\+', r'\1', clean_text)  # +underline+ -> underline
        clean_text = re.sub(r'-([^-]*)-', r'\1', clean_text)    # -strikethrough- -> strikethrough
        clean_text = re.sub(r'\^([^^]*)\^', r'\1', clean_text)  # ^superscript^ -> superscript
        clean_text = re.sub(r'~([^~]*)~', r'\1', clean_text)    # ~subscript~ -> subscript
        
        # Remove code blocks and inline code
        clean_text = re.sub(r'\{\{([^{}]*)\}\}', r'\1', clean_text)  # {{code}} -> code
        clean_text = re.sub(r'`([^`]*)`', r'\1', clean_text)        # `code` -> code
        
        # Remove headers (h1., h2., etc.)
        clean_text = re.sub(r'^h[1-6]\.\s*', '', clean_text, flags=re.MULTILINE)
        
        # Remove quote blocks (bq.)
        clean_text = re.sub(r'^bq\.\s*', '', clean_text, flags=re.MULTILINE)
        
        # Remove list markers
        clean_text = re.sub(r'^\*+\s*', '', clean_text, flags=re.MULTILINE)  # Bullet lists
        clean_text = re.sub(r'^#+\s*', '', clean_text, flags=re.MULTILINE)   # Numbered lists
        clean_text = re.sub(r'^-+\s*', '', clean_text, flags=re.MULTILINE)   # Dash lists
        
        # Clean up multiple spaces and normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Trim whitespace
        clean_text = clean_text.strip()
        
        return clean_text
        
    def detect_language(self, text: str) -> str:
        """
        Detect language of text using langdetect
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr', etc.)
        """
        try:
            # Clean text for better detection
            clean_text = str(text).strip().replace('\n', ' ').replace('\r', '')
            if len(clean_text) < 10:  # Skip very short texts
                return 'unknown'
            
            detected_lang = detect(clean_text)
            return detected_lang
        except (LangDetectException, Exception):
            return 'unknown'
    
    def translate_text(self, text: str, source_lang: str = 'auto') -> Dict[str, str]:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            source_lang: Source language code
            
        Returns:
            Dictionary with original text, detected language, and translation
        """
        try:
            # Clean the text first
            cleaned_text = self.clean_jira_text(text)
            
            # Skip if already English, unknown, or too long
            if source_lang in ['en', 'unknown'] or len(cleaned_text) > 4500:
                return {
                    'original_text': text,
                    'detected_language': source_lang,
                    'translated_text': cleaned_text,  # Use cleaned version
                    'translation_needed': False
                }
            
            # Translate using cleaned text
            translated = self.translator.translate(cleaned_text)
            
            return {
                'original_text': text,
                'detected_language': source_lang,
                'translated_text': translated,
                'translation_needed': True
            }
            
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return {
                'original_text': text,
                'detected_language': source_lang,
                'translated_text': text,
                'translation_needed': False,
                'error': str(e)
            }
    
    def get_user_input(self) -> Dict[str, str]:
        """
        Collect user input for query parameters
        
        Returns:
            Dictionary with user-provided parameters
        """
        print("=== GTSE Query Translator ===")
        print("This tool will query BigQuery and translate any non-English content found.\n")
        
        # Get query file name
        query_file = input("Enter the query file name (e.g., 'query.txt', 'cmp_query.txt'): ").strip()
        if not query_file:
            query_file = 'query.txt'  # Default
        
        # Get date range
        print("\nDate Range (leave blank to skip date filtering):")
        start_date = input("Start date (YYYY-MM-DD): ").strip()
        end_date = input("End date (YYYY-MM-DD): ").strip()
        
        # Validate dates if provided
        if start_date or end_date:
            try:
                if start_date:
                    datetime.strptime(start_date, '%Y-%m-%d')
                if end_date:
                    datetime.strptime(end_date, '%Y-%m-%d')
                if start_date and end_date and start_date > end_date:
                    print("Warning: Start date is after end date!")
            except ValueError:
                print("Invalid date format! Please use YYYY-MM-DD format.")
                return self.get_user_input()  # Retry
        
        # Get text columns to translate
        print("\nEnter the column names containing text to translate:")
        print("You can enter multiple columns separated by commas (e.g., 'description,comments,notes')")
        text_columns_input = input("Column names: ").strip()
        
        if not text_columns_input:
            text_columns = ['description']  # Default
        else:
            # Split by comma and clean up whitespace
            text_columns = [col.strip() for col in text_columns_input.split(',') if col.strip()]
        
        print(f"Will translate columns: {text_columns}")
        
        # Ask about additional parameters
        use_group_json = input("\nDo you want to load a group list from JSON? (y/n): ").strip().lower() == 'y'
        group_json_config = None
        
        if use_group_json:
            json_file = input("JSON file path: ").strip()
            print("JSON path (dot-separated, e.g., 'allMembersInfo.allMembersInGroup'):")
            json_path_str = input("Path: ").strip()
            json_path = json_path_str.split('.') if json_path_str else ['allMembersInfo', 'allMembersInGroup']
            
            group_json_config = {
                'file': json_file,
                'path': json_path
            }
        
        # Ask about custom parameters
        use_custom_params = input("\nDo you want to add custom parameters? (y/n): ").strip().lower() == 'y'
        custom_parameters = {}
        
        if use_custom_params:
            print("Enter custom parameters (format: name=value, press Enter with empty name to finish):")
            while True:
                param_input = input("Parameter: ").strip()
                if not param_input:
                    break
                if '=' in param_input:
                    name, value = param_input.split('=', 1)
                    custom_parameters[name.strip()] = value.strip()
                else:
                    print("Invalid format. Use: name=value")
        
        return {
            'query_file': query_file,
            'start_date': start_date if start_date else None,
            'end_date': end_date if end_date else None,
            'text_columns': text_columns,
            'group_json_config': group_json_config,
            'custom_parameters': custom_parameters if custom_parameters else None
        }
    
    def process_translation_batch(self, df: pd.DataFrame, text_columns: List[str], batch_size: int = 50) -> pd.DataFrame:
        """
        Process translations using optimized two-pass approach
        
        Args:
            df: DataFrame with query results
            text_columns: List of column names containing text to translate
            batch_size: Number of rows to process at once (not used in two-pass)
            
        Returns:
            DataFrame with translation results added
        """
        if df.empty:
            self.logger.warning("No data to translate")
            return df
        
        # Check which columns exist
        available_columns = list(df.columns)
        valid_columns = [col for col in text_columns if col in available_columns]
        missing_columns = [col for col in text_columns if col not in available_columns]
        
        if missing_columns:
            self.logger.warning(f"Columns not found in query results: {missing_columns}")
        
        if not valid_columns:
            self.logger.error(f"None of the specified text columns found in query results")
            self.logger.info(f"Available columns: {available_columns}")
            return df
        
        self.logger.info(f"Starting optimized two-pass translation of {len(df)} rows for columns: {valid_columns}")
        
        # Initialize result columns for each text column
        for col in valid_columns:
            df[f'{col}_detected_language'] = ''
            df[f'{col}_translated_text'] = ''
            df[f'{col}_translation_needed'] = False
        
        # PASS 1: Language Detection and Cleaning (Fast Pass)
        self.logger.info("Pass 1: Detecting languages and cleaning text...")
        translation_queue = []  # List of (row_idx, col, cleaned_text, detected_lang)
        
        for idx, row in df.iterrows():
            for col in valid_columns:
                text = str(row[col])
                
                # Skip empty or very short text
                if pd.isna(text) or len(str(text).strip()) < 3:
                    df.at[idx, f'{col}_detected_language'] = 'empty'
                    df.at[idx, f'{col}_translated_text'] = text
                    df.at[idx, f'{col}_translation_needed'] = False
                    continue
                
                # Clean text once
                cleaned_text = self.clean_jira_text(text)
                
                # Smart English detection - skip language detection for obviously English text
                sample_text = cleaned_text[:200].lower()  # Check first 200 chars
                if (re.match(r'^[a-zA-Z\s\.\,\!\?\-\(\)0-9\:\;\&\%\$\#\@\+\=\_\[\]\"\'\/\\]+$', sample_text) and
                    len([word for word in sample_text.split() if word in ['the', 'and', 'or', 'is', 'was', 'are', 'were', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'this', 'that', 'error', 'stage', 'system', 'problem', 'issue']]) >= 2):
                    detected_lang = 'en'  # Skip expensive language detection
                else:
                    # Detect language for non-obvious cases
                    detected_lang = self.detect_language(cleaned_text)
                
                # Store results immediately
                df.at[idx, f'{col}_detected_language'] = detected_lang
                df.at[idx, f'{col}_translated_text'] = cleaned_text  # Always store cleaned version
                
                # Queue for translation if needed
                if detected_lang not in ['en', 'unknown', 'empty'] and len(cleaned_text) <= 4500:
                    translation_queue.append((idx, col, cleaned_text, detected_lang))
                    df.at[idx, f'{col}_translation_needed'] = True
                else:
                    df.at[idx, f'{col}_translation_needed'] = False
        
        # PASS 2: Actual Translation (Only for non-English)
        total_to_translate = len(translation_queue)
        self.logger.info(f"Pass 2: Translating {total_to_translate} non-English text fields...")
        
        if total_to_translate == 0:
            self.logger.info("No translations needed - all text is already in English!")
            return df
        
        for i, (idx, col, cleaned_text, detected_lang) in enumerate(translation_queue, 1):
            try:
                # Translate the cleaned text directly
                translated = self.translator.translate(cleaned_text)
                df.at[idx, f'{col}_translated_text'] = translated
                
                # Progress update at milestones (every 10% or every 500 translations, whichever is more frequent)
                milestone_interval = max(50, min(500, total_to_translate // 10))  # 10% intervals, but at least every 50, max every 500
                if i % milestone_interval == 0 or i == total_to_translate:
                    progress = (i / total_to_translate) * 100
                    self.logger.info(f"Translation progress: {i}/{total_to_translate} ({progress:.1f}%)")
                
                # Rate limiting - smaller delay since we're only translating what's needed
                time.sleep(0.02)  # Reduced from 0.05 to 0.02
                
            except Exception as e:
                self.logger.error(f"Translation error for row {idx}, col {col}: {e}")
                # Keep the cleaned text if translation fails
                continue
        
        self.logger.info(f"Translation complete! Processed {len(df)} rows, translated {total_to_translate} fields")
        return df
    
    def get_translation_summary(self, df: pd.DataFrame, text_columns: List[str]) -> Dict:
        """
        Generate summary statistics for translation results across multiple columns
        
        Args:
            df: DataFrame with translation results
            text_columns: List of text columns that were translated
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {'total_rows': 0, 'message': 'No data to summarize'}
        
        total_rows = len(df)
        summary = {
            'total_rows': total_rows,
            'columns_processed': [],
            'overall_stats': {}
        }
        
        # Find which columns were actually processed
        translation_columns = [col for col in text_columns if f'{col}_translation_needed' in df.columns]
        
        if not translation_columns:
            summary['message'] = 'No translation columns found in results'
            return summary
        
        summary['columns_processed'] = translation_columns
        
        # Overall statistics across all columns
        all_translations_needed = []
        all_detected_languages = []
        
        # Per-column statistics
        column_stats = {}
        
        for col in translation_columns:
            translation_needed_col = f'{col}_translation_needed'
            detected_language_col = f'{col}_detected_language'
            
            if translation_needed_col in df.columns:
                translated_rows = df[df[translation_needed_col] == True]
                
                column_stats[col] = {
                    'translated_rows': len(translated_rows),
                    'percentage_translated': (len(translated_rows) / total_rows * 100) if total_rows > 0 else 0,
                }
                
                # Collect data for overall stats
                all_translations_needed.extend(df[translation_needed_col].tolist())
                
                if detected_language_col in df.columns:
                    languages = df[detected_language_col].value_counts().to_dict()
                    column_stats[col]['languages_detected'] = languages
                    all_detected_languages.extend(df[detected_language_col].tolist())
                    
                    # Top non-English languages for this column
                    if not translated_rows.empty:
                        column_stats[col]['top_non_english_languages'] = translated_rows[detected_language_col].value_counts().head(5).to_dict()
        
        summary['column_stats'] = column_stats
        
        # Overall statistics
        total_translations_needed = sum(all_translations_needed)
        total_possible_translations = len(all_translations_needed)
        
        summary['overall_stats'] = {
            'total_text_fields': total_possible_translations,
            'total_translations_needed': total_translations_needed,
            'overall_translation_percentage': (total_translations_needed / total_possible_translations * 100) if total_possible_translations > 0 else 0
        }
        
        # Overall language distribution
        if all_detected_languages:
            language_counts = pd.Series(all_detected_languages).value_counts()
            summary['overall_stats']['all_languages_detected'] = language_counts.to_dict()
            
            # Filter out English and empty for non-English summary
            non_english_languages = language_counts[~language_counts.index.isin(['en', 'empty', 'unknown'])]
            if not non_english_languages.empty:
                summary['overall_stats']['top_non_english_languages'] = non_english_languages.head(5).to_dict()
        
        return summary
    
    def save_results(self, df: pd.DataFrame, base_filename: str = 'gtse_translated_results') -> str:
        """
        Save results to CSV with timestamp
        
        Args:
            df: DataFrame to save
            base_filename: Base name for the output file
            
        Returns:
            Filename of saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_filename}_{timestamp}.csv"
        
        try:
            df.to_csv(filename, index=False)
            self.logger.info(f"Results saved to: {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return None
    
    def run_interactive_query(self):
        """
        Run the complete interactive query and translation workflow
        """
        try:
            # Get user input
            params = self.get_user_input()
            
            print(f"\n=== Executing Query ===")
            print(f"Query file: {params['query_file']}")
            if params['start_date'] and params['end_date']:
                print(f"Date range: {params['start_date']} to {params['end_date']}")
            print(f"Text columns to translate: {params['text_columns']}")
            
            # Execute BigQuery
            results = self.bigquery_client.execute_query(
                query_file=params['query_file'],
                start_date=params['start_date'],
                end_date=params['end_date'],
                group_json_config=params['group_json_config'],
                custom_parameters=params['custom_parameters']
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            print(f"\n=== Query Results ===")
            print(f"Retrieved {len(df)} rows from BigQuery")
            
            if df.empty:
                print("No data returned from query.")
                return
            
            print(f"Columns available: {list(df.columns)}")
            
            # Process translations
            print(f"\n=== Translation Processing ===")
            df_translated = self.process_translation_batch(df, params['text_columns'])
            
            # Generate summary
            summary = self.get_translation_summary(df_translated, params['text_columns'])
            
            print(f"\n=== Translation Summary ===")
            print(f"Total rows: {summary['total_rows']}")
            
            if 'overall_stats' in summary:
                overall = summary['overall_stats']
                print(f"Total text fields processed: {overall.get('total_text_fields', 0)}")
                print(f"Total translations needed: {overall.get('total_translations_needed', 0)}")
                print(f"Overall translation percentage: {overall.get('overall_translation_percentage', 0):.1f}%")
                
                if 'all_languages_detected' in overall:
                    print(f"\nAll languages detected:")
                    for lang, count in overall['all_languages_detected'].items():
                        print(f"  {lang}: {count} occurrences")
            
            if 'column_stats' in summary:
                print(f"\n=== Per-Column Statistics ===")
                for col, stats in summary['column_stats'].items():
                    print(f"\nColumn '{col}':")
                    print(f"  Rows translated: {stats['translated_rows']}")
                    print(f"  Percentage translated: {stats['percentage_translated']:.1f}%")
                    
                    if 'languages_detected' in stats:
                        print(f"  Languages in this column:")
                        for lang, count in list(stats['languages_detected'].items())[:5]:  # Show top 5
                            print(f"    {lang}: {count} rows")
            
            # Save results
            filename = self.save_results(df_translated)
            
            # Show sample translations
            print(f"\n=== Sample Translations ===")
            samples_shown = 0
            max_samples = 3
            
            for col in params['text_columns']:
                translation_needed_col = f'{col}_translation_needed'
                if translation_needed_col in df_translated.columns:
                    non_english_samples = df_translated[df_translated[translation_needed_col] == True].head(2)
                    
                    if not non_english_samples.empty:
                        print(f"\nColumn '{col}' samples:")
                        for _, row in non_english_samples.iterrows():
                            if samples_shown >= max_samples:
                                break
                                
                            detected_lang = row.get(f'{col}_detected_language', 'unknown')
                            original_text = str(row[col])
                            translated_text = str(row.get(f'{col}_translated_text', ''))
                            
                            print(f"  Language: {detected_lang}")
                            print(f"  Original: {original_text[:150]}...")
                            print(f"  Translation: {translated_text[:150]}...")
                            print("-" * 60)
                            samples_shown += 1
                        
                        if samples_shown >= max_samples:
                            break
            
            if samples_shown == 0:
                print("No non-English content found to translate.")
            
            print(f"\n=== Complete ===")
            if filename:
                print(f"Results saved to: {filename}")
            print("Translation process completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in interactive query: {e}")
            print(f"Error occurred: {e}")
            raise

def main():
    """Main entry point"""
    try:
        translator = GTSEQueryTranslator()
        translator.run_interactive_query()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()