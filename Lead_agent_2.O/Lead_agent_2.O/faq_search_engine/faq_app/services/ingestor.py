import pandas as pd
import os
from typing import Optional
from faq_app.utils.logger import get_logger

logger = get_logger("DataIngestor")

class DataIngestor:
    COLUMN_MAPPING = {
        'question': ['q', 'query', 'problem', 'issue', 'question text'],
        'answer': ['a', 'solution', 'resolution', 'response', 'answer text']
    }

    @classmethod
    def normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names to 'question' and 'answer'."""
        df.columns = [str(c).lower().strip() for c in df.columns]
        rename_map = {}
        
        for standard, variations in cls.COLUMN_MAPPING.items():
            if standard in df.columns:
                continue
            for v in variations:
                if v in df.columns:
                    rename_map[v] = standard
                    break
        
        return df.rename(columns=rename_map)

    @classmethod
    def load_file(cls, file_path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(file_path):
            return None
            
        ext = file_path.split('.')[-1].lower()
        try:
            if ext in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            elif ext == 'csv':
                # Robust CSV reading
                df = pd.read_csv(
                    file_path,  
                    on_bad_lines='skip', 
                    encoding='utf-8',
                    encoding_errors='replace'
                )
            else:
                logger.warning(f"Unsupported format: {file_path}")
                return None

            df = cls.normalize_columns(df)
            
            df = cls.normalize_columns(df)
            
            # Validation
            required = ['question', 'answer']
            if not all(col in df.columns for col in required):
                # Fallback: Try reading without header if it looks like the known 4-column format
                # This handles cases where users upload "Category,Topic,Question,Answer" data without the header row
                try:
                    if ext == 'csv':
                        df_no_header = pd.read_csv(
                            file_path,  
                            on_bad_lines='skip', 
                            encoding='utf-8',
                            encoding_errors='replace',
                            header=None
                        )
                        # Check if it has 4 columns (Category, Topic, Question, Answer)
                        if len(df_no_header.columns) == 4:
                            logger.info(f"Detected potential headerless CSV with 4 columns: {file_path}")
                            df_no_header.columns = ['category', 'topic', 'question', 'answer']
                            df = df_no_header
                            # Re-validate
                            if all(col in df.columns for col in required):
                                logger.info(f"Successfully recovered headerless CSV: {file_path}")
                except Exception as e:
                    logger.warning(f"Fallback CSV read failed: {e}")

            if not all(col in df.columns for col in required):
                logger.warning(f"Skipping {file_path}: Missing required columns.")
                return None
            
            # Remove rows with empty question or answer
            df = df.dropna(subset=required)
            
            # Preserve all columns for new format support (id, normalized_questions, aliases, etc.)
            # Convert required columns to string, keep others as-is
            for col in required:
                df[col] = df[col].astype(str)
            
            return df

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {str(e)}")
            return None
