"""
Data Pipeline Module
Handles ingestion, cleaning, and unification of all three datasets:
1. Entry Level Career QA Dataset
2. CareerVillage Q&A Dataset  
3. HR Interview Questions Dataset
"""
import json
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Generator, Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FAQ_CORPUS_FILE

# Maximum number of HR Interview records to load (to prevent memory issues)
MAX_HR_RECORDS = 50000


def stream_json_file(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Stream JSON file line by line for memory efficiency.
    Handles both JSON arrays and JSON lines format.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # Try parsing as JSON array first
        if content.startswith('['):
            data = json.loads(content)
            for item in data:
                yield item
        else:
            # JSON lines format
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def extract_zip(zip_path: Path, extract_to: Path) -> list:
    """Extract zip file and return list of extracted file paths."""
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        extracted_files = [extract_to / name for name in zip_ref.namelist()]
    return extracted_files


def load_entry_level_career_qa(data_dir: Path) -> pd.DataFrame:
    """
    Load Entry Level Career QA Dataset.
    Expected format: CSV with 'role', 'question', 'answer' columns.
    """
    print("📂 Loading Entry Level Career QA Dataset...")
    
    zip_path = data_dir / "Entry Level Career QA Dataset.zip"
    if not zip_path.exists():
        # Check in parent directory
        zip_path = data_dir.parent.parent / "Entry Level Career QA Dataset.zip"
    
    if not zip_path.exists():
        print(f"⚠️ Entry Level Career QA Dataset not found at {zip_path}")
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    # Extract zip
    extract_dir = data_dir / "entry_level_qa"
    extract_dir.mkdir(exist_ok=True)
    
    # Only extract if not already extracted
    if not any(extract_dir.glob("*.csv")):
        extract_zip(zip_path, extract_dir)
    
    # Find and load CSV files
    dfs = []
    for csv_file in extract_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            print(f"  ✓ Loaded {csv_file.name}: {len(df)} rows")
            
            # This dataset has columns: role, question, answer
            if 'question' in df.columns and 'answer' in df.columns:
                df = df[['question', 'answer']].copy()
                dfs.append(df)
            else:
                # Try to find matching columns
                col_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'question' in col_lower or 'query' in col_lower:
                        col_mapping[col] = 'question'
                    elif 'answer' in col_lower or 'response' in col_lower:
                        col_mapping[col] = 'answer'
                
                if col_mapping:
                    df = df.rename(columns=col_mapping)
                    if 'question' in df.columns and 'answer' in df.columns:
                        df = df[['question', 'answer']].copy()
                        dfs.append(df)
                        
        except Exception as e:
            print(f"  ✗ Error loading {csv_file.name}: {e}")
    
    if not dfs:
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['source'] = 'entry_level_career_qa'
    
    print(f"  ✓ Total Entry Level records: {len(combined_df)}")
    return combined_df


def load_careervillage_data(data_dir: Path) -> pd.DataFrame:
    """
    Load CareerVillage Q&A Dataset.
    This dataset has questions.csv and answers.csv that need to be joined.
    """
    print("📂 Loading CareerVillage Dataset...")
    
    zip_path = data_dir / "data-science-for-good-careervillage.zip"
    if not zip_path.exists():
        zip_path = data_dir.parent.parent / "data-science-for-good-careervillage.zip"
    
    if not zip_path.exists():
        print(f"⚠️ CareerVillage Dataset not found at {zip_path}")
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    # Extract zip
    extract_dir = data_dir / "careervillage"
    extract_dir.mkdir(exist_ok=True)
    extract_zip(zip_path, extract_dir)
    
    # Find questions and answers files (only these two specific files)
    questions_df = None
    answers_df = None
    
    for csv_file in extract_dir.rglob("*.csv"):
        filename = csv_file.name.lower()
        try:
            # Only load the main questions.csv and answers.csv files
            if filename == 'questions.csv':
                questions_df = pd.read_csv(csv_file)
                print(f"  ✓ Loaded {csv_file.name}: {len(questions_df)} rows")
            elif filename == 'answers.csv':
                answers_df = pd.read_csv(csv_file)
                print(f"  ✓ Loaded {csv_file.name}: {len(answers_df)} rows")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file.name}: {e}")
    
    if questions_df is None or answers_df is None:
        print("⚠️ Could not find both questions.csv and answers.csv files")
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    # CareerVillage specific column names
    # Questions: questions_id, questions_title, questions_body
    # Answers: answers_question_id, answers_body
    
    # Combine title and body for better question text
    if 'questions_title' in questions_df.columns and 'questions_body' in questions_df.columns:
        questions_df['question_text'] = questions_df['questions_title'].fillna('') + ' ' + questions_df['questions_body'].fillna('')
    elif 'questions_title' in questions_df.columns:
        questions_df['question_text'] = questions_df['questions_title']
    elif 'questions_body' in questions_df.columns:
        questions_df['question_text'] = questions_df['questions_body']
    
    # Clean HTML from answers
    if 'answers_body' in answers_df.columns:
        # Simple HTML tag removal
        answers_df['answer_text'] = answers_df['answers_body'].str.replace(r'<[^>]+>', ' ', regex=True)
        answers_df['answer_text'] = answers_df['answer_text'].str.replace(r'\s+', ' ', regex=True)
    
    # Merge questions and answers
    print("  🔄 Merging questions and answers...")
    merged = pd.merge(
        questions_df[['questions_id', 'question_text']],
        answers_df[['answers_question_id', 'answer_text']],
        left_on='questions_id',
        right_on='answers_question_id',
        how='inner'
    )
    
    # Rename columns
    merged = merged.rename(columns={
        'question_text': 'question',
        'answer_text': 'answer'
    })
    
    # Keep only relevant columns
    merged = merged[['question', 'answer']]
    merged['source'] = 'careervillage'
    
    print(f"  ✓ Merged dataset: {len(merged)} Q&A pairs")
    return merged


def load_hr_interview_data(data_dir: Path, max_records: int = MAX_HR_RECORDS) -> pd.DataFrame:
    """
    Load HR Interview Questions and Ideal Answers Dataset.
    Uses streaming JSON parsing for memory efficiency.
    
    Args:
        data_dir: Directory containing raw data
        max_records: Maximum number of records to load (default: 50000)
    """
    print("📂 Loading HR Interview Questions Dataset...")
    
    zip_path = data_dir / "HR Interview Questions and Ideal Answers.zip"
    if not zip_path.exists():
        zip_path = data_dir.parent.parent / "HR Interview Questions and Ideal Answers.zip"
    
    if not zip_path.exists():
        print(f"⚠️ HR Interview Dataset not found at {zip_path}")
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    # Extract zip
    extract_dir = data_dir / "hr_interview"
    extract_dir.mkdir(exist_ok=True)
    extract_zip(zip_path, extract_dir)
    
    records = []
    
    # Process JSON files with ijson streaming (memory-efficient)
    for json_file in extract_dir.rglob("*.json"):
        print(f"  📄 Streaming {json_file.name} (max {max_records} records)...")
        
        try:
            # Try using ijson for true streaming
            try:
                import ijson
                with open(json_file, 'rb') as f:
                    # Parse items from JSON array
                    parser = ijson.items(f, 'item')
                    for item in tqdm(parser, desc="Reading JSON", total=max_records):
                        if len(records) >= max_records:
                            print(f"  ℹ️ Reached max records limit ({max_records})")
                            break
                        
                        record = extract_qa_from_item(item)
                        if record:
                            records.append(record)
                            
            except ImportError:
                # Fallback: Read file in chunks and parse manually
                print("  ℹ️ ijson not installed, using chunked reading...")
                records.extend(parse_json_chunked(json_file, max_records))
                
        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
            # Try CSV fallback
            pass
    
    # Also check for CSV files
    for csv_file in extract_dir.rglob("*.csv"):
        if len(records) >= max_records:
            break
        try:
            df = pd.read_csv(csv_file, nrows=max_records - len(records))
            print(f"  ✓ Loaded {csv_file.name}: {len(df)} rows")
            
            # Standardize column names
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'question' in col_lower or 'query' in col_lower:
                    col_mapping[col] = 'question'
                elif 'answer' in col_lower or 'response' in col_lower or 'ideal' in col_lower:
                    col_mapping[col] = 'answer'
            
            df = df.rename(columns=col_mapping)
            if 'question' in df.columns and 'answer' in df.columns:
                records.extend(df[['question', 'answer']].to_dict('records'))
        except Exception as e:
            print(f"  ✗ Error loading {csv_file.name}: {e}")
    
    if not records:
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    df = pd.DataFrame(records)
    df['source'] = 'hr_interview'
    
    print(f"  ✓ Total HR Interview records: {len(df)}")
    return df


def extract_qa_from_item(item: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Extract question and answer from a JSON item."""
    record = {}
    
    # HR Interview JSON has specific keys: 'question' and 'ideal_answer'
    if 'question' in item:
        record['question'] = str(item['question'])
    if 'ideal_answer' in item:
        record['answer'] = str(item['ideal_answer'])
    
    # Fallback: try to find matching keys
    if 'question' not in record or 'answer' not in record:
        for key in item.keys():
            key_lower = key.lower()
            if 'question' not in record and ('question' in key_lower or 'query' in key_lower):
                record['question'] = str(item[key])
            elif 'answer' not in record and ('answer' in key_lower or 'response' in key_lower or 'ideal' in key_lower):
                record['answer'] = str(item[key])
    
    if 'question' in record and 'answer' in record:
        # Clean up: remove the question repeated in the answer (seen in some records)
        if record['answer'].endswith(record['question']):
            record['answer'] = record['answer'][:-len(record['question'])].strip()
        return record
    return None


def parse_json_chunked(json_file: Path, max_records: int) -> list:
    """
    Parse JSON file by reading chunks - fallback when ijson isn't available.
    """
    records = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        # Read first few MB to check structure
        content = f.read(10 * 1024 * 1024)  # 10MB
        
        # Check if it's a JSON array
        if content.strip().startswith('['):
            # Parse items from the chunk
            depth = 0
            current_item = ""
            in_string = False
            escape = False
            started = False
            
            for char in content:
                if escape:
                    escape = False
                    if started:
                        current_item += char
                    continue
                if char == '\\':
                    escape = True
                    if started:
                        current_item += char
                    continue
                if char == '"':
                    in_string = not in_string
                
                if not in_string:
                    if char == '{':
                        if depth == 0:
                            started = True
                            current_item = char
                        else:
                            current_item += char
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if started:
                            current_item += char
                        if depth == 0 and started:
                            try:
                                item = json.loads(current_item)
                                record = extract_qa_from_item(item)
                                if record:
                                    records.append(record)
                                    if len(records) >= max_records:
                                        return records
                            except json.JSONDecodeError:
                                pass
                            current_item = ""
                            started = False
                    elif started:
                        current_item += char
                elif started:
                    current_item += char
    
    return records


def unify_datasets() -> pd.DataFrame:
    """
    Load all three datasets and unify them into a single FAQ corpus.
    """
    print("\n" + "="*60)
    print("🔄 UNIFYING ALL DATASETS")
    print("="*60 + "\n")
    
    # Load each dataset
    entry_level_df = load_entry_level_career_qa(RAW_DATA_DIR)
    careervillage_df = load_careervillage_data(RAW_DATA_DIR)
    hr_interview_df = load_hr_interview_data(RAW_DATA_DIR)
    
    # Combine all datasets
    print("\n📊 Combining datasets...")
    
    dfs_to_combine = []
    
    if not entry_level_df.empty:
        dfs_to_combine.append(entry_level_df[['question', 'answer', 'source']])
        print(f"  ✓ Entry Level Career QA: {len(entry_level_df)} records")
    
    if not careervillage_df.empty:
        dfs_to_combine.append(careervillage_df[['question', 'answer', 'source']])
        print(f"  ✓ CareerVillage: {len(careervillage_df)} records")
    
    if not hr_interview_df.empty:
        dfs_to_combine.append(hr_interview_df[['question', 'answer', 'source']])
        print(f"  ✓ HR Interview: {len(hr_interview_df)} records")
    
    if not dfs_to_combine:
        print("❌ No data loaded from any dataset!")
        return pd.DataFrame(columns=['question', 'answer', 'source'])
    
    unified_df = pd.concat(dfs_to_combine, ignore_index=True)
    
    # Clean the data
    print("\n🧹 Cleaning data...")
    unified_df = unified_df.dropna(subset=['question', 'answer'])
    unified_df['question'] = unified_df['question'].astype(str).str.strip()
    unified_df['answer'] = unified_df['answer'].astype(str).str.strip()
    
    # Remove empty strings
    unified_df = unified_df[unified_df['question'].str.len() > 0]
    unified_df = unified_df[unified_df['answer'].str.len() > 0]
    
    # Remove duplicates
    initial_count = len(unified_df)
    unified_df = unified_df.drop_duplicates(subset=['question'], keep='first')
    print(f"  ✓ Removed {initial_count - len(unified_df)} duplicate questions")
    
    # Add unique ID
    unified_df = unified_df.reset_index(drop=True)
    unified_df['faq_id'] = range(len(unified_df))
    
    # Reorder columns
    unified_df = unified_df[['faq_id', 'question', 'answer', 'source']]
    
    print(f"\n✅ Final unified corpus: {len(unified_df)} Q&A pairs")
    
    return unified_df


def save_corpus(df: pd.DataFrame) -> Path:
    """Save the unified corpus to CSV."""
    output_path = PROCESSED_DATA_DIR / FAQ_CORPUS_FILE
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved corpus to: {output_path}")
    return output_path


def run_pipeline() -> pd.DataFrame:
    """Run the complete data pipeline."""
    print("\n" + "🚀"*20)
    print("   FAQ DATA PIPELINE")
    print("🚀"*20 + "\n")
    
    # Unify datasets
    corpus = unify_datasets()
    
    if corpus.empty:
        print("\n❌ Pipeline failed: No data was loaded")
        return corpus
    
    # Save to CSV
    save_corpus(corpus)
    
    # Print summary
    print("\n" + "="*60)
    print("📈 PIPELINE SUMMARY")
    print("="*60)
    print(f"Total Q&A pairs: {len(corpus)}")
    print(f"\nBy source:")
    print(corpus['source'].value_counts().to_string())
    print("\nSample questions:")
    for i, row in corpus.head(3).iterrows():
        print(f"  {i+1}. {row['question'][:80]}...")
    print("="*60 + "\n")
    
    return corpus


if __name__ == "__main__":
    run_pipeline()

