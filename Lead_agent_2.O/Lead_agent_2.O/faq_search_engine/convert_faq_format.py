"""
Script to convert FAQ CSV files to the new format:
id,question,normalized_questions,answer,audience,category,aliases,normalized_allias
"""
import pandas as pd
import re
import os
from pathlib import Path

def normalize_text(text):
    """Normalize text to lowercase and remove extra spaces."""
    if pd.isna(text) or not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def generate_id(prefix, index):
    """Generate unique ID for FAQ entry."""
    return f"{prefix}_faq_{index}"

def categorize_question(question):
    """Categorize question based on keywords."""
    question_lower = question.lower()
    
    # Define category mappings
    categories = {
        'engagement_models_working_style': ['engagement', 'model', 'working', 'style', 'onboard', 'delivery'],
        'services_offerings': ['service', 'offer', 'provide', 'develop', 'build', 'specialize', 'solution'],
        'company_info': ['found', 'founded', 'headquarter', 'employee', 'revenue', 'country', 'year', 'founder'],
        'blockchain': ['blockchain', 'defi', 'nft', 'smart contract', 'crypto', 'dapp', 'decentralized'],
        'ai_ml': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'predictive', 'chatbot'],
        'mobile_development': ['mobile', 'app', 'ios', 'android', 'react native', 'flutter'],
        'cloud_devops': ['cloud', 'aws', 'azure', 'gcp', 'devops', 'migration', 'ci/cd'],
        'iot': ['iot', 'internet of things', 'device', 'sensor'],
        'metaverse_gaming': ['metaverse', 'ar', 'vr', 'gaming', 'game', 'virtual'],
        'security_compliance': ['security', 'encryption', 'compliance', 'gdpr', 'hipaa', 'audit'],
        'integration': ['integrate', 'integration', 'api', 'payment', 'gateway', 'crm'],
        'general': []  # Default category
    }
    
    for category, keywords in categories.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
    
    return 'general'

def process_faq_file(input_path, output_path, prefix="EnterpriseAI"):
    """Convert FAQ file to new format."""
    print(f"Processing: {input_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return
    
    # Normalize column names
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    # Check for required columns
    if 'question' not in df.columns or 'answer' not in df.columns:
        print(f"Skipping {input_path}: Missing required columns (question, answer)")
        return
    
    # Remove duplicates based on question
    df = df.drop_duplicates(subset=['question'], keep='first')
    
    # Create new dataframe with required columns
    new_rows = []
    
    for idx, row in df.iterrows():
        question = str(row['question']).strip()
        answer = str(row['answer']).strip()
        
        if not question or not answer or question == 'nan' or answer == 'nan':
            continue
        
        # Generate ID
        faq_id = generate_id(prefix, idx + 1)
        
        # Normalize question
        normalized_question = normalize_text(question)
        
        # Categorize
        category = categorize_question(question)
        
        # Set audience (default to 'public')
        audience = 'public'
        
        # For now, aliases are empty (can be populated later)
        aliases = ""
        normalized_aliases = ""
        
        new_rows.append({
            'id': faq_id,
            'question': question,
            'normalized_questions': normalized_question,
            'answer': answer,
            'audience': audience,
            'category': category,
            'aliases': aliases,
            'normalized_allias': normalized_aliases
        })
    
    # Create new dataframe
    new_df = pd.DataFrame(new_rows)
    
    # Save to new file
    new_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[OK] Converted {len(new_rows)} FAQs to {output_path}")
    print(f"  Categories: {new_df['category'].value_counts().to_dict()}\n")

def main():
    """Main conversion function."""
    base_dir = Path(__file__).parent / "data"
    
    # Process tech_company/faq.csv
    tech_company_dir = base_dir / "tech_company"
    if (tech_company_dir / "faq.csv").exists():
        # Backup original
        backup_path = tech_company_dir / "faq_backup.csv"
        if not backup_path.exists():
            import shutil
            shutil.copy(tech_company_dir / "faq.csv", backup_path)
            print(f"Backed up original to {backup_path}")
        
        process_faq_file(
            tech_company_dir / "faq.csv",
            tech_company_dir / "faq.csv",  # Replace original
            prefix="EnterpriseAI"
        )
    
    # Process tech_customer/faq.csv
    tech_customer_dir = base_dir / "tech_customer"
    if (tech_customer_dir / "faq.csv").exists():
        # Backup original
        backup_path = tech_customer_dir / "faq_backup.csv"
        if not backup_path.exists():
            import shutil
            shutil.copy(tech_customer_dir / "faq.csv", backup_path)
            print(f"Backed up original to {backup_path}")
        
        process_faq_file(
            tech_customer_dir / "faq.csv",
            tech_customer_dir / "faq.csv",  # Replace original
            prefix="EnterpriseAI"
        )
    
    # Process default directory files
    default_dir = base_dir / "default"
    if default_dir.exists():
        for csv_file in default_dir.glob("*.csv"):
            # Skip backup files and already converted files
            if "_new" in csv_file.name or "_backup" in csv_file.name:
                continue
            output_file = default_dir / f"{csv_file.stem}_new.csv"
            process_faq_file(
                csv_file,
                output_file,
                prefix="faq"
            )
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()


